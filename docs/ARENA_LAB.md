# Arena lab

The arena lab is an interactive behavioral crowd experiment with 50–500 independent agents in a 64 × 48 m model venue. It uses engineered routing and movement rules, not a population of simulated connectome brains. The separate maze/connectome experiment remains available on its own server. See [the connectome scaling plan](CONNECTOME_SCALING.md) for that remaining work.

## Run

From the repository with its Python dependencies installed:

```powershell
python scripts/arena_demo.py --population 200 --seed 7 --scenario open
```

Open **http://127.0.0.1:8766/**. The service binds only to loopback. It does not publish anything externally. The maze server can continue independently at http://127.0.0.1:8765/.

For the prepared local workspace, `Start-Arena-Demo.ps1` beside this guide starts the service if needed and opens the browser. It reuses a running arena service after checking its model identity. Source-only installations use the Python command above.

## Using the arena

1. Set **50–500 agents**, choose a repeatable seed, and select **Apply & reset**. The default is 200 agents.
2. Choose one of four interface presets: **Open arena** (four exits), **Exit closure** (two exits closed), **Crowd surge** (faster preferred movement and shorter reaction delays), or **Repellent plume** (open layout with an active aversive source).
3. Select **Start evacuation**. Inspect route choices and queues, then click an exit row to open or close that exit during the run. Closing every exit is allowed and can leave agents unable to leave.
4. Change wind direction and speed, adjust repellent strength, or choose **Place source in arena** and click walkable space. The source is restricted to the actual walkable grid; stage, seating blocks and exterior space are excluded. The field follows the same venue geometry used by agents.
5. Click a fly to inspect its speed, behavioral state, chosen exit and cumulative model exposure. Enable density shading, recent trails, wind particles or velocity arrows. Zoom, fit and fullscreen controls change only the view.
6. Pause, change playback speed, reset or export the experiment. Reset reconstructs the seeded population; live exit changes do not reset the run.

The initial population stays inside until the alarm. Gathering agents avoid the local repellent concentration gradient through an explicitly engineered rule; evacuation routes also account for the field. The browser requests simulation steps, so pausing stops world time. **Repellent plume** is a frontend preset combining the backend's `open` scenario with repellent strength 1.5; CLI scenarios remain `open`, `blocked` and `surge`.

## Measurements and export

The interface reports agents remaining, percentage evacuated, a rolling five-second exit rate, peak local density within a two-metre radius, experiment time and elapsed evacuation time. The evacuation curve uses time since the alarm. Exit counts show route distribution; overlap and slow/stalled-agent readouts expose numerical quality and congestion.

The inspector's exposure is accumulated model concentration over time, with no biological or toxicological units. Density shading is a visual accumulation overlay rather than a calibrated safety threshold.

**Export experiment** downloads JSON containing the seed and configuration, model description, final/current agent state, metrics, sampled run history, recorded control events and environment. It is an observation record, not a complete frame-by-frame replay or neural checkpoint.

## Interpretation

- The agents use a behavioral movement model. Their apparent urgency does not establish fly neural behavior or human emergency behavior.
- Agents have independent seeded behavioral state, response delays, preferred speed and route choices. They do not learn neural policies in this arena.
- Collision/contact handling uses illustrative inertia, mass-weighted contact correction, tangential friction and swept grid checks. Numerical overlap can remain and is displayed. This is not a model of crushing, injury, falling or human contact pressure.
- The fly drawings use **model-scale footprints**, approximately 0.48 m in diameter, to make contact and crowd flow legible in the venue. They are not life-size fruit flies or a geometrically faithful animal embodiment.
- Hazard concentration and exposure are model quantities, not measurements of smoke toxicity or safe exposure limits.
- Results are suitable for inspecting this model, comparing runs and finding implementation problems. They are not validated evacuation or building-safety advice.
- Wind particles and density shading illustrate computed fields. They are not additional physical agents and do not steer the crowd.

## Local API

| Request | Input/result |
|---|---|
| `GET /api/state` | Current agents, exits, model metadata, metrics and computation timing |
| `GET /api/environment` | Arena geometry and current hazard/environment grid |
| `POST /api/step` | `{ "steps": 1, "dt": 0.05 }`; 1–10 steps, 0.005–0.05 seconds each |
| `POST /api/reset` | Configuration overrides: population, seed, preset; scenario is an alias for preset |
| `POST /api/control` | One action: alarm, exit, preset, hazard strength/location, or wind dials |
| `GET /api/export` | Download the experiment JSON |
| `POST /api/export` | Same export with an empty JSON object |

Examples of controls:

```json
{"alarm":true}
{"exit":{"id":"an ID returned by /api/state","open":false}}
{"wind_direction_deg":180,"wind_speed_mps":1.3}
{"hazard_strength":1}
```

POST requests require `Content-Type: application/json`. Inputs are bounded and the session serializes requests. A reset that fails to construct a valid simulation preserves the previous session.

`performance.compute_rtf` means simulated time advanced divided by time spent computing steps. It excludes idle/browser time and is not rendering FPS. It must be measured at the intended population and scenario before making real-time capacity claims.

The service intentionally hosts one experiment. It is not a public authenticated multi-user application.
