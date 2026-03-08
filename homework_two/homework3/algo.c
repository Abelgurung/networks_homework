#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <net/tcp.h>

// very basic algorithm : 


// half the window after congestion
static u32 algo_ssthresh(struct sock *sk)
{
	const struct tcp_sock *tp = tcp_sk(sk);
	return max(tp->snd_cwnd >> 1U, 2U);
}


static void algo_cong_avoid(struct sock *sk, u32 ack, u32 acked)
{
	struct tcp_sock *tp = tcp_sk(sk);

	if (!tcp_is_cwnd_limited(sk))
		return;

	//slowstart if below threshhold
	if (tp->snd_cwnd < tp->snd_ssthresh) {
		acked = tcp_slow_start(tp, acked);
		if (!acked)
			return;
	}

	// additive increase - congestion-avoidance phase
	tcp_cong_avoid_ai(tp, tp->snd_cwnd, acked);
}

// needed
static u32 algo_undo_cwnd(struct sock *sk)
{
	return tcp_sk(sk)->snd_cwnd;
}

static struct tcp_congestion_ops algo __read_mostly = {
	.name       = "algo",
	.owner      = THIS_MODULE,
	.ssthresh   = algo_ssthresh,
	.cong_avoid = algo_cong_avoid,
	.undo_cwnd  = algo_undo_cwnd,
};

static int __init algo_init(void)
{
	return tcp_register_congestion_control(&algo);
}

static void __exit algo_exit(void)
{
	tcp_unregister_congestion_control(&algo);
}

module_init(algo_init);
module_exit(algo_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Rachit");
MODULE_DESCRIPTION("Simple TCP congestion control Algorithm");