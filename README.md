# Real-Time Systems - Unit 4 Lab (Mini Projects 4.1 - 4.3)

University of Rwanda, Gako Campus - Computer Engineering, Year 4 - Group assignment

Members: Janvier NSHIMIYIMANA (223026679), Blandon ASHIMIRWE (223026675), Gilbert HIRWA NTWALI (223026689)

The written report is `Lab4_RTS_Report.pdf`. All simulations are pure Python 3.10+ (standard library plus matplotlib) and are seeded, so results are reproducible.

| Folder | Content | Run |
|---|---|---|
| `Mini-Project-4.1` | `rt_sampling_sim.py`: sampling jitter and processing overruns in a fixed-period control loop; sweep table, figures, 200-400 word summary | `python rt_sampling_sim.py --sweep` (or `--jitter 0.3 --overrun-prob 0.05`) |
| `Mini-Project-4.2` | `env_pipeline.py`: multi-rate monitoring pipeline, FP / EDF / round-robin, compliance reports, figures, report | `python env_pipeline.py --all` (or `--rule RR --streams 4`) |
| `Mini-Project-4.3` | `edge_cloud_sim.py`: edge vs cloud vs hybrid latency, congestion sweep, figures, report | `python edge_cloud_sim.py --all` |

```
pip install -r requirements.txt
```
Latency models, processing times and signal shapes are modelling assumptions stated in the code docstrings and in the report. `*_output.txt` and `*.csv` files are the output of the last run.
