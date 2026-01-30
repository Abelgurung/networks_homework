cargo -Z unstable-options -C pinger build --release
./pinger/target/release/pinger list.txt | tee data.txt

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

pip install -r requirements.txt

python3 q1.py data.txt
python3 q2.py list.txt
