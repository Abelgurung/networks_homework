#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <net/tcp.h>

// algorithm : 
// increase window by 1 per RTT when RTT is stable 
// if RTT rises, stop increasing
// if packet loss, half the window

#define RTT_BUFFER 105
#define RTT_WINDOW 5

// circular buffer
struct algo {
	u32 rtt_hist[RTT_WINDOW];
	u8 rtt_idx;
	u8 rtt_cnt;
};

static void algo_init(struct sock *sk)
{
	// pointer to space in socket 
	struct algo *ca = inet_csk_ca(sk);
	int i;

	// clear history- to make sure every new flow gets a fresh state
	for (i = 0; i < RTT_WINDOW; i++)
		ca->rtt_hist[i] = 0;

	ca->rtt_idx = 0;
	ca->rtt_cnt = 0;
}

// half the window after congestion
static u32 algo_ssthresh(struct sock *sk)
{
	const struct tcp_sock *tp = tcp_sk(sk);
	return max(tp->snd_cwnd >> 1U, 2U);
}

static u32 algo_windowed_avg_rtt(const struct algo *ca)
{
	u64 summ = 0;
	int i;

	if (ca->rtt_cnt == 0)
		return 0;

	for (i = 0; i < ca->rtt_cnt; i++)
		summ += ca->rtt_hist[i];

	return (u32)(summ / ca->rtt_cnt);
}

static void algo_add_rtt_sample(struct algo *ca, u32 srtt_us)
{
	// srtt is smoothed rtt stored in linux
	if (srtt_us == 0)
		return;

	ca->rtt_hist[ca->rtt_idx] = srtt_us;
	ca->rtt_idx = (ca->rtt_idx + 1) % RTT_WINDOW;

	if (ca->rtt_cnt < RTT_WINDOW)
		ca->rtt_cnt++;
}


static void algo_cong_avoid(struct sock *sk, u32 ack, u32 acked)
{
	struct tcp_sock *tp = tcp_sk(sk);
	struct algo *ca = inet_csk_ca(sk);
	u32 srtt_us, avg_rtt_us;

	if (!tcp_is_cwnd_limited(sk))
		return;

	// for some reason, it's stored like this in linux (3 bit shifted)
	srtt_us = tp->srtt_us >> 3;

	// update window
	algo_add_rtt_sample(ca, srtt_us);
	avg_rtt_us = algo_windowed_avg_rtt(ca);

	//slowstart if below threshhold
	if (tp->snd_cwnd < tp->snd_ssthresh) {
		acked = tcp_slow_start(tp, acked);
		if (!acked)
			return;
	}

	// Grow only if current RTT is still stable relative to the average RTT over the last 5 samples.
	if (avg_rtt_us == 0 || srtt_us == 0 || srtt_us * 100 <= avg_rtt_us * RTT_BUFFER) {
		// additive increase - congestion-avoidance phase
		tcp_cong_avoid_ai(tp, tp->snd_cwnd, acked);
	} else {
		return;
	}

}

// needed
static u32 algo_undo_cwnd(struct sock *sk)
{
	return tcp_sk(sk)->snd_cwnd;
}

// kernel interface
static struct tcp_congestion_ops algo __read_mostly = {
	.init		= algo_init,
	.name       = "algo",
	.owner      = THIS_MODULE,
	.ssthresh   = algo_ssthresh,
	.cong_avoid = algo_cong_avoid,
	.undo_cwnd  = algo_undo_cwnd,
};

static int __init algo_register(void)
{
	return tcp_register_congestion_control(&algo);
}

static void __exit algo_unregister(void)
{
	tcp_unregister_congestion_control(&algo);
}

module_init(algo_register);
module_exit(algo_unregister);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Rachit");
MODULE_DESCRIPTION("TCP congestion control Algorithm");