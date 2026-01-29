use std::{collections::BTreeMap, io::BufRead as _, str::FromStr as _, sync::Arc};

use clap::Parser;

#[derive(Parser, Debug)]
struct Arguments {
    list: clio::Input,
}

const PING_COUNT: std::num::NonZeroU16 = std::num::NonZeroU16::new(100).unwrap();

fn main() {
    let mut arguments = Arguments::parse();

    let addrs: BTreeMap<_, _> = arguments
        .list
        .lock()
        .lines()
        .map(|l| l.unwrap())
        .inspect(|s| assert_eq!(s, s.trim()))
        .filter_map(|s| get_ip(&s).map(|a| (s, a)))
        .collect();

    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(run(&addrs))
}

fn get_ip(location: &str) -> Option<std::net::Ipv4Addr> {
    if let Ok(val) = std::net::Ipv4Addr::from_str(location) {
        Some(val)
    } else {
        let lookup = match dns_lookup::lookup_host(location) {
            Ok(iter) => iter,
            Err(e) => panic!("dns lookup error {e} on {location:?}"),
        };
        let addresses: Vec<_> = lookup
            .filter_map(|a| match a {
                std::net::IpAddr::V4(ipv4_addr) => Some(ipv4_addr),
                _ => None,
            })
            .collect();
        match addresses.len() {
            0 => None,
            1 => Some(addresses[0]),
            _ => panic!("too many addresses {addresses:?}"),
        }
    }
}

async fn run(addresses: &BTreeMap<String, std::net::Ipv4Addr>) {
    let workers = Arc::new(tokio::sync::Semaphore::new(4));

    let mut pool = tokio::task::JoinSet::new();
    for (name, addr) in addresses {
        let name = name.clone();
        let addr = *addr;
        let workers = workers.clone();
        pool.spawn(async move {
            (
                name,
                ping_server(std::net::IpAddr::V4(addr), PING_COUNT, workers).await,
            )
        });
    }

    let statistics: BTreeMap<_, _> = pool.join_all().await.into_iter().collect();

    for name in addresses.keys() {
        println!("{name}: {:?}", statistics.get(name).unwrap());
    }
}

async fn ping_server(
    address: std::net::IpAddr,
    pings: std::num::NonZeroU16,
    permitter: Arc<tokio::sync::Semaphore>,
) -> Result<Statistics, surge_ping::SurgeError> {
    let mut min = std::time::Duration::MAX;
    let mut max = std::time::Duration::ZERO;
    let mut total_time = std::time::Duration::ZERO;

    for _ in 0..u16::from(pings) {
        // up to 4.295 seconds * 4
        tokio::time::sleep(std::time::Duration::from_nanos(
            u64::from(rand::random::<u32>()) * 4,
        ))
        .await;

        let payload = rand::random::<u64>().to_le_bytes();

        let permit = permitter.acquire().await;
        let (_packet, duration) = surge_ping::ping(address, &payload).await?;
        drop(permit);

        if duration < min {
            min = duration;
        }
        if duration > max {
            max = duration;
        }
        total_time += duration;
    }

    Ok(Statistics {
        address,
        min,
        max,
        avg: total_time / u32::from(u16::from(pings)),
    })
}

#[derive(Debug)]
#[allow(unused)]
struct Statistics {
    address: std::net::IpAddr,
    min: std::time::Duration,
    max: std::time::Duration,
    avg: std::time::Duration,
}
