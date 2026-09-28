# fly-net · Fly Pong

Play Pong against a fruit fly's brain. The fly's paddle (top) is driven by **1,370 spiking neurons wired exactly as in the FlyWire whole-brain connectome**. The ball excites the fly's object-detecting eye neurons (LC10a). The fly's own wiring turns that into activity in its steering neurons (DNa02/DNa01), and their right-minus-left firing moves the paddle. Nothing about its play is scripted. You can also train it with dopamine-style plasticity. The fly's only input is the ball: every 10 ms, its direction and apparent size from the fly's paddle set the firing rates of the 234 eye neurons. It cannot see your paddle, the walls or the score. The only output is four steering neurons: turn = (right DNa02 − left DNa02) + ½ (right DNa01 − left DNa01), smoothed over 25 ms, sets the paddle's speed.

```
ball ─► LC10a eye neurons (234) ─► AOTU019 / AOTU025 ─► … 1,128 more neurons … ─► DNa02 / DNa01 ─► paddle
         receptive fields from       central / peripheral     every neuron this        steering        speed = 0.045 × (R − L)
         the wiring itself           pursuit relays           input can reach          neurons
```

## By the numbers

| | Count | What it is |
|---|---|---|
| Model input | 2 numbers | the ball's direction and apparent size, seen from the fly's paddle, updated every 10 ms |
| Input neurons | 234 | LC10a eye neurons; the ball sets how fast each one fires |
| Output neurons | 4 | left and right DNa02 and DNa01; their right-minus-left firing moves the paddle |
| Neurons in all | 1,370 | the 234 and the 4 above, plus 4 relay neurons and 1,128 others that the ball can make fire |
| Connections | 82,282 | the model's weights. Each is FlyWire's synapse count × 0.275 mV, excitatory or inhibitory by transmitter; none was tuned to play Pong |
| Trainable connections | 10,866 | every connection out of the 234 input neurons. Training scales each one by 0–3×; the other 71,416 never change |
| Hand-set parameters | 26 | numbers that aren't in the wiring: 9 for the neuron model (from Shiu et al.) and 17 we chose (eyes 6, steering readout 4, learning 7). The game lists them all under "26 hand-set parameters" |

## Play

**Live: https://kaiwong-sapiens.github.io/fly-net/**

Or open `game/index.html` in a browser. It works straight from disk, because the circuit loads as a script from `game/data/`, or you can serve the folder with `python3 -m http.server -d game`.

- **You play**: your paddle is at the bottom. On a phone, slide your thumb along the pad under the arena (or drag on the arena); on a computer, move the mouse or use ← → / A D.
- **Naive fly / Courting fly**: eye gain 45 vs 120 spikes/s. Courting males turn up LC10a gain to chase a female (Hindmarsh Sten et al. 2021).
- **Train it**: *Coach* gives sugar while the fly steers toward the ball and heat while it steers away. *Rally reward* gives sugar per return and heat per miss. You can also press *Give sugar* / *Give heat* yourself. *Fast practice* runs 60 s of fly-vs-wall training as fast as your machine allows.
- Hover or tap any neuron on the brain map to see its type, transmitter and FlyWire ID. The brain map, eye view and steering meters are drawn from the front, as you face the fly, so its right side is on your left.

## Results

Return rate of the fly's paddle against a wall that sends back random angles. Each row is 3 runs × 5 minutes (`pipeline/experiments/verify_stats.py`, log in `results/verify_stats.log`).

| Fly | Balls returned |
|---|---|
| Courting fly (eye gain 120/s), connectome weights | **80%** (78–81) |
| Naive fly (eye gain 45/s), connectome weights | **36%** (35–37) |
| Motionless paddle | **24%** (23–25) |
| Naive fly after 2 min of coaching, tested with learning off | **84%** (82–86) |

Coaching changes about 2,500 of the 10,866 LC10a output connections, almost all strengthened. The median connection is untouched.

**What the wiring does on its own** (`results/steering_sweep.log`). While the ball is within about ±30° of straight ahead, it drives the central relay AOTU019, but the steering neurons stay nearly silent (under 16 spikes/s). Further out, the peripheral relay AOTU025 joins in and that side's DNa02 fires hard, reaching 60–110 spikes/s at 90°. The opposite DNa02 stays silent. So the fly holds still while the ball is roughly in front and swings when it drifts to the side. The wiring explains why: AOTU025 is excitatory and drives the DNa02 on its own side (154 and 167 synapses), while AOTU019 is GABAergic and inhibits the DNa02 on the opposite side (216 and 121 synapses), so a ball in front, which drives both AOTU019s, holds both steering neurons down. That matches the two parallel pursuit pathways described by Collie et al. (2026).

**Rally reward barely works; the coach does** (`pipeline/experiments/learning_rules.py`). With sugar per return and heat per miss, the synapse changes cancel out. A miss is usually a turn that came too late, and punishing the synapses that were active makes the next turn weaker still. Continuous feedback on steering direction works within two minutes. That is the logic of the classic flight-simulator conditioning experiments.

## How it's built (and checked)

| Step | Script | Check |
|---|---|---|
| Whole-brain LIF simulator: every neuron identical, weight = synapse count × transmitter sign × 0.275 mV, parameters of Shiu et al. 2024 | `pipeline/flysim.py` (numba) | vs the published Brian2 model (sugar-neuron activation): **r = 0.999**, mean difference 0.6 spikes/s (`validate_brian2.py`; the published run shows 404 neurons active over 30 trials, ours 384 over 10, because rarely-firing neurons appear less often in fewer trials) |
| Find what drives steering: stimulate each visual projection type on one side | `pipeline/probe_lc.py` | LC10a → ipsilateral DNa02 (the courtship-pursuit pathway); LPLC2 → giant fibre DNp01 (escape) |
| Give each LC10a a receptive field from the wiring: dendrite position = input-weighted centroid of its columnar eye inputs, oriented by its AOTU019 (central) vs AOTU025 (peripheral) preference | `pipeline/retina.py` | the anatomical axis predicts that preference at **r = 0.90 / 0.93** (right / left eye), and the same on held-out neurons |
| Keep only neurons the game can ever make fire, including after maximal training (LC10a synapses ×3) | `pipeline/build_subnet.py` | 1,370 of 138,639 neurons |
| Sub-circuit vs whole brain | `pipeline/validate_subnet.py` | **identical** steering-neuron spike trains on 6 random ball trajectories with random synapse strengths |
| Export for the browser | `pipeline/export_subnet.py` | 1,370 neurons, 82,282 connections, 1.06 MB |
| Closed-loop reference game + learning rule | `pipeline/pong_sim.py` | the browser engine reproduces its steering rates (e.g. DNa02 9.2 spikes/s at −30° in both) |

**In the browser** the spiking engine runs in a Web Worker, so drawing and touch never wait on it; on a slow phone the game slows down rather than freezing the page, and without Worker support it falls back to the page with at most 8 ms of simulation per frame. Each 0.1 ms step visits only neurons that hold charge and targets with spikes in transit, which is spike-for-spike identical to updating all 1,370 neurons (checked on 30 s of play).

One Brian2 detail mattered and is copied in `flysim.py` and the browser engine: **synaptic input that arrives while a neuron is in its 2.2 ms refractory period is discarded.** Brian2 does this automatically for variables flagged `unless refractory`. Without it, strongly driven neurons fire about 20% too fast.

### Taken from the data vs modelled

- **Data (FlyWire v783):** who connects to whom, synapse counts, excitatory/inhibitory sign from predicted transmitters (GABA and glutamate inhibit, everything else excites, as in Shiu et al.), cell types, and each LC10a's place on the eye map.
- **Modelled:**
  - the neuron model;
  - the ball → LC10a mapping (Gaussian receptive fields 16° wide; each eye assumed to see from 15° across the midline to 150° to its side, and from 60° below to 70° above);
  - the steering → paddle readout: DNa01 counted at half the weight of DNa02, a gain of 0.045 field widths/s per spike/s, and 25 ms smoothing. Varying each (`pipeline/experiments/readout_sweep.py`, `results/readout_sweep.log`): DNa01's weight makes no difference, because it barely fires; a higher gain makes the fly better (courting fly 47% at 0.015, 80% at 0.045, 91% at 0.1); 25 ms smoothing plays better than 10 ms (72%) or 50 ms (77%);
  - the learning rule: a three-factor rule where an eligibility trace (τ = 0.8 s) of pre×post coincidences is turned into change by dopamine, clipped to 0–3× the wired strength, and applied to LC10a output connections. This is the mushroom body's form of plasticity, and applying it here is our assumption;
  - "arousal" as an input gain. P1 neurons raise LC10a gain in courting males, but this is a female brain with no P1.

## Deploy

The site is served by GitHub Pages from the `gh-pages` branch, which holds a copy of `game/`. After changing the game on `main`, publish it with:

```bash
git subtree push --prefix game origin gh-pages
```

## Reproduce

```bash
./get_data.sh                          # FlyWire v783 connectivity (Shiu et al. repo) + FlyWire annotations → data/
pip install -r requirements.txt
python pipeline/validate_brian2.py     # simulator vs published Brian2 results (add --brian2 /path/to/python for a live Brian2 run; needs numpy<2)
python pipeline/build_subnet.py        # ~3 min
python pipeline/validate_subnet.py
python pipeline/export_subnet.py       # writes game/data/flypong_circuit.js and work/flypong_circuit.json
python pipeline/experiments/verify_stats.py
python pipeline/experiments/readout_sweep.py   # how much the readout's choices matter
```

## The male brain (from the Google/Janelia post)

`pipeline/load_malecns.py` loads the male CNS connectome from neuPrint in the same format. It takes every neuron within 3 synaptic steps downstream of LC10a, and it places LC10a receptive fields using their actual input-synapse locations in the lobula. Then run the same steps with `--dataset malecns`:

```bash
pip install neuprint-python
export NEUPRINT_TOKEN=...            # neuprint.janelia.org → Account
export MALECNS_DATASET=male-cns:v0.9 # check client.fetch_datasets() for the current name
python pipeline/build_subnet.py --dataset malecns
python pipeline/validate_subnet.py --dataset malecns
python pipeline/export_subnet.py --dataset malecns
```

**Untested.** neuPrint was unreachable from the environment this was written in, so property names (`consensusNt`/`predictedNt`, `somaSide`, `superclass`) and axis orientation need checking on first run. The male brain is the interesting one for this game: it has P1 neurons, which gate the LC10a pursuit pathway during courtship, and dimorphic AOTU neurons such as AOTU008.

## More games this circuit could host

- **Dodgeball**: looming balls excite LPLC2 → giant fibre (DNp01); the fly jumps. Can training teach it not to flinch?
- **Feed the fly**: sugar vs bitter taste neurons → proboscis motor neuron MN9, the flagship result of Shiu et al.
- **Odour casino**: a two-choice learning game on the mushroom body using the real projection-neuron → Kenyon-cell wiring.
- **Courtship chase**: the male CNS with P1 arousal switched on by the player.

## Layout

```
game/index.html                 the game: engine (spiking brain, eyes, learning, physics) in a Web Worker; the page only draws
game/data/flypong_circuit.js    exported sub-circuit
pipeline/                       simulator, receptive fields, extraction, validation, export, closed-loop reference
pipeline/experiments/           steering sweep, learning-rule comparison, return-rate verification
results/                        logs behind the numbers above
```

## Credits

- Dorkenwald et al. 2024, *Neuronal wiring diagram of an adult brain*, Nature; Schlegel et al. 2024, *Whole-brain annotation and multi-connectome cell typing of Drosophila*, Nature. FlyWire data and annotations via [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations).
- Shiu et al. 2024, *A Drosophila computational brain model reveals sensorimotor processing*, Nature. Model parameters, connectivity files and reference results via [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) (MIT).
- Collie et al. 2026, *Specialized parallel pathways for adaptive control of visual object pursuit*, Neuron (LC10a → AOTU019/AOTU025 → DNa02).
- Rayshubskiy et al. 2025, *Neural circuit mechanisms for steering control in walking Drosophila*, eLife (DNa02/DNa01 steering).
- Hindmarsh Sten et al. 2021, *Sexual arousal gates visual processing during Drosophila courtship*, Nature (LC10a gain).
- Male CNS connectome: *Sexual dimorphism in the complete connectome of the Drosophila male central nervous system*, Cell 2026 ([Google Research post](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/)).
