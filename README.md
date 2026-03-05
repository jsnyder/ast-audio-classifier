# AST Audio Classifier

Real-time audio event detection for Home Assistant using the [Audio Spectrogram Transformer](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) (AST) model. Listens to RTSP camera audio streams and publishes detected sound events (dog bark, smoke alarm, glass break, etc.) as binary sensors via MQTT auto-discovery.

## Features

- **AST classification** across 527 AudioSet labels, grouped into 28 meaningful categories
- **CLAP zero-shot verification** using [LAION CLAP](https://huggingface.co/laion/clap-htsat-fused) to confirm, suppress, or override AST predictions
- **MQTT auto-discovery** creates Home Assistant binary sensors automatically
- **Multi-camera support** with per-camera thresholds, cooldowns, and high-pass filtering
- **Adaptive thresholds** that adjust to ambient noise levels
- **Noise stress scoring** tracks cumulative audio activity per camera
- **Event consolidation** deduplicates detections across overlapping cameras
- **OpenObserve logging** for event analytics and debugging
- **Optional LLM judge** for sampled clip evaluation via any OpenAI-compatible API
- **Pre-trigger audio buffer** captures 500ms before the detection for better clip context
- **Air-gapped compatible** with models pre-downloaded at build time

## How It Works

1. Connects to RTSP audio streams from your cameras (via FFmpeg)
2. Buffers 3-second audio clips when sound exceeds a configurable dB threshold
3. Runs the AST model to classify the audio into one of 28 groups
4. Optionally verifies with CLAP zero-shot classification
5. Publishes detections to MQTT, creating Home Assistant binary sensors

## Installation

### As a Home Assistant Add-on

Add this repository to your Home Assistant add-on store, then install and configure via the add-on UI.

### As a Standalone Docker Container

```bash
docker build -t ast-audio-classifier .
docker run -d \
  -v /path/to/config.yaml:/data/config.yaml \
  -p 8080:8080 \
  ast-audio-classifier
```

## Configuration

Create a `config.yaml` file:

```yaml
mqtt:
  host: core-mosquitto
  port: 1883
  username: "${MQTT_USERNAME}"
  password: "${MQTT_PASSWORD}"

cameras:
  - name: living_room
    rtsp_url: "rtsp://192.168.1.100:8554/living_room"
    db_threshold: -35.0
    cooldown_seconds: 10
  - name: backyard
    rtsp_url: "rtsp://192.168.1.100:8554/backyard"
    db_threshold: -30.0
    battery: true
    reconnect_interval: 60

# Optional: adaptive thresholds
  - name: kitchen
    rtsp_url: "rtsp://192.168.1.100:8554/kitchen"
    adaptive_threshold: true
    adaptive_margin_db: 8.0
    highpass_freq: 120

defaults:
  confidence_threshold: 0.15
  auto_off_seconds: 30
  clip_duration_seconds: 3
```

String values support `${ENV_VAR}` substitution for secrets.

### CLAP Verification (Optional)

CLAP provides a second opinion on AST detections, reducing false positives:

```yaml
clap:
  enabled: true
  confirm_threshold: 0.30
  suppress_threshold: 0.15
  override_threshold: 0.40
  discovery_threshold: 0.50
  confirm_margin: 0.20
  never_suppress:
    - smoke_alarm
    - glass_break
  custom_prompts:
    vacuum_cleaner:
      - "a robot vacuum cleaner running"
      - "a vacuum cleaner motor humming"
```

### OpenObserve Logging (Optional)

```yaml
openobserve:
  host: "192.168.1.50"
  port: 5080
  org: "default"
  stream: "ast_audio"
  username: "${OO_USERNAME}"
  password: "${OO_PASSWORD}"
```

### LLM Judge (Optional)

Sends sampled audio clips to a multimodal LLM for ground-truth evaluation:

```yaml
llm_judge:
  enabled: true
  api_base: "https://your-openai-compatible-endpoint/v1"
  api_key: "${LLM_JUDGE_API_KEY}"
  model: "gemini-2.5-flash"
  sample_rate: 0.10
```

`api_base` is required when `enabled: true`. Any OpenAI-compatible API endpoint works (OpenRouter, LiteLLM, etc.).

## Detection Groups

The 527 AudioSet labels are mapped to these binary sensor groups:

| Group | Examples |
|-------|----------|
| `dog_bark` | Bark, Growling, Whimper |
| `cat_meow` | Meow, Purr, Hiss, Cat |
| `smoke_alarm` | Smoke detector, Fire alarm |
| `glass_break` | Shatter, Breaking glass |
| `doorbell` | Ding-dong, Doorbell |
| `speech` | Speech, Conversation, Narration |
| `crying` | Crying, Baby cry, Wail |
| `music` | Music, Singing, Musical instrument |
| `siren` | Siren, Emergency vehicle |
| `gunshot` | Gunshot, Explosion |
| ... | 18 more groups |

Each group becomes a Home Assistant binary sensor, e.g., `binary_sensor.ast_audio_classifier_living_room_dog_bark`.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with model status and camera states |
| `/classify` | POST | Upload a WAV file for classification |
| `/status` | GET | Detailed diagnostics |

## Development

```bash
# Set up environment
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Install PyTorch CPU-only
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Run tests
pytest

# Lint
ruff check src/ tests/
```

## Models & Licenses

| Component | Model | License |
|-----------|-------|---------|
| Primary classifier | [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) | MIT |
| CLAP verifier | [laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused) | Apache 2.0 |
| AudioSet labels | [Google AudioSet](https://research.google.com/audioset/) | CC BY 4.0 |

## License

[MIT](LICENSE)
