# Enterprise Knowledge Assistant (NVIDIA GPU)

> A production-grade, local-first RAG system optimized for NVIDIA GPUs

## Overview

Enterprise Knowledge Assistant is a complete Retrieval-Augmented Generation (RAG) system that runs entirely on your local machine with NVIDIA GPU acceleration. Built with Hugging Face's Text Generation Inference (TGI), it leverages your GPU's CUDA cores to provide fast, accurate answers based exclusively on your internal documentation.

### Key Features

- **Zero Hallucination Guarantee** - Answers only from provided context
- **Full Source Attribution** - Every response includes document title, URL, and version
- **GPU-Accelerated** - Runs 100% locally using NVIDIA CUDA via TGI
- **Complete Observability** - Integrated Prometheus, Loki, and Grafana monitoring
- **Flexible Data Sources** - Confluence, web scraping, or mock data for testing

## Prerequisites

Before you begin, ensure you have:

- **Docker Desktop** - Latest version installed and running
- **NVIDIA GPU** - Compatible GPU with CUDA support (GTX 1060 or better recommended)
- **NVIDIA Container Toolkit** - [Installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- **16GB+ System RAM** - For optimal performance with Mistral-7B
- **10GB+ Disk Space** - For model storage and containers

### Verifying GPU Setup

Confirm your GPU is accessible to Docker:

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

You should see your GPU information displayed.

## Quick Start

### 1. Configure Environment

Clone the repository and set up your configuration:

```bash
cd enterprise-rag-local
cp .env.example .env
```

Edit `.env` and choose your data source:

**Option A: Test Mode (Recommended First)**
```bash
INGEST_MODE=mock
TGI_MODEL=mistralai/Mistral-7B-Instruct-v0.2
```

**Option B: Web Scraping**
```bash
INGEST_MODE=online
ONLINE_DATASOURCES=https://example.com/docs,https://another-site.com/wiki
```

**Option C: Confluence Integration**
```bash
INGEST_MODE=confluence
CONFLUENCE_BASE_URL=https://your-company.atlassian.net
CONFLUENCE_USERNAME=your-email@company.com
CONFLUENCE_API_TOKEN=your-api-token
CONFLUENCE_SPACES=SPACE1,SPACE2
```

### 2. Launch the System

Build and start all services:

```bash
docker-compose up -d --build
```

### 3. Monitor Initial Setup

The first launch takes 5-10 minutes to download the LLM model (approximately 4-5GB). Watch progress:

```bash
# Monitor TGI model download and initialization
docker-compose logs -f tgi

# Wait for "Connected" or "Serving model" message
```

Once TGI is ready, monitor data ingestion:

```bash
# Monitor data ingestion (wait for "sync completed")
docker-compose logs -f ingest
```

### 4. Start Using

Once ingestion completes, access the system:

- **Chat Interface**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **TGI Health**: http://localhost:8080/health
- **Grafana Dashboard**: http://localhost:3000 (admin/admin)

## System Components

### Core Services

| Service | Purpose | Port | GPU |
|---------|---------|------|-----|
| **ui** | Streamlit chat interface | 8501 | No |
| **api** | FastAPI orchestration layer | 8000 | No |
| **tgi** | Text Generation Inference (LLM) | 8080 | Yes |
| **qdrant** | Vector database | 6333 | No |
| **ingest** | Data ingestion pipeline | - | No |

### Monitoring Stack

| Service | Purpose | Port |
|---------|---------|------|
| **grafana** | Visualization dashboard | 3000 |
| **prometheus** | Metrics collection | 9090 |
| **loki** | Log aggregation | 3100 |
| **promtail** | Log shipping | - |

### Initialization Jobs

- **init-db**: One-time vector database collection setup
- **ingest**: One-time data processing and embedding generation

## Project Structure

```
enterprise-rag-local/
├── .env                      # Your configuration
├── .env.example              # Configuration template
├── README.md                 # This file
├── ARCHITECTURE.md           # System design documentation
├── docker-compose.yml        # Service orchestration
├── requirements.txt          # Python dependencies
│
├── docker/                   # Container definitions
│   ├── Dockerfile.api
│   ├── Dockerfile.base
│   ├── Dockerfile.ingest
│   └── Dockerfile.ui
│
├── loki/
│   └── loki-config.yml
├── prometheus/
│   └── prometheus.yml
├── promtail/
│   └── promtail-config.yml
│
└── src/                      # Application code
    ├── config.py             # Settings management
    ├── data_pipeline.py      # Confluence ingestion
    ├── init_db.py            # Database setup
    ├── main_api.py           # API orchestrator
    ├── main_ui.py            # Streamlit UI
    ├── mock_data.py          # Test data
    ├── mock_pipeline.py      # Mock ingestion
    ├── online_pipeline.py    # Web scraping
    └── run_ingestion.py      # Ingestion entrypoint
```

## GPU Configuration

### TGI Model Options

Edit `.env` to use different Hugging Face models:

```bash
# Default - Mistral 7B (Recommended)
TGI_MODEL=mistralai/Mistral-7B-Instruct-v0.2

# Alternative - Llama 2 7B
TGI_MODEL=meta-llama/Llama-2-7b-chat-hf

# Larger model - Mistral 22B (Requires more VRAM)
TGI_MODEL=mistralai/Mistral-22B-Instruct-v0.1

# Smaller model - Phi-2 (For GPUs with limited VRAM)
TGI_MODEL=microsoft/phi-2
```

**Note**: Some models require Hugging Face authentication. Set `HUGGING_FACE_HUB_TOKEN` in `.env` if needed.

### TGI Parameters

Adjust TGI launch parameters in `docker-compose.yml`:

```yaml
tgi:
  command:
    - --model-id=${TGI_MODEL}
    - --max-batch-prefill-tokens=4096
    - --max-total-tokens=8192
    - --max-input-length=4096
    - --max-concurrent-requests=128
```

**Common Adjustments:**
- `--quantize=bitsandbytes-nf4` - Reduce VRAM usage (4-bit quantization)
- `--num-shard=1` - Number of GPUs to use
- `--max-batch-size=8` - Concurrent request batching

### GPU Memory Requirements

| Model | Minimum VRAM | Recommended VRAM |
|-------|--------------|------------------|
| Phi-2 (2.7B) | 4GB | 6GB |
| Mistral 7B | 8GB | 12GB |
| Llama 2 13B | 16GB | 24GB |
| Mistral 22B | 24GB | 32GB |

## Monitoring & Observability

### Accessing Grafana

1. Navigate to http://localhost:3000
2. Login with `admin` / `admin`
3. Configure data sources:
   - **Prometheus**: http://prometheus:9090
   - **Loki**: http://loki:3100

### Key Metrics

- `questions_total` - Total questions asked
- `questions_duration_seconds` - Response latency
- `retrieval_chunks_found` - Context chunks retrieved
- `llm_failures_total` - TGI service errors
- `tgi_request_duration` - GPU inference time
- `tgi_queue_size` - Pending requests

### GPU Monitoring

Monitor GPU utilization:

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Check TGI GPU metrics
curl http://localhost:8080/metrics | grep gpu
```

### Log Queries

Access logs via Grafana Explore:

```
{container_name="tgi"}
{container_name="api"}
{container_name=~"api|ingest"} |= "error"
{container_name="tgi"} |= "CUDA"
```

## Troubleshooting

### TGI Service Unavailable (503)

The TGI service may still be initializing or has encountered an error:

```bash
# Check TGI logs
docker-compose logs -f tgi

# Look for "Connected" message

# Restart if needed
docker-compose restart tgi
```

### CUDA Out of Memory

If TGI crashes with OOM errors:

1. **Use a smaller model** in `.env`
2. **Enable quantization** in `docker-compose.yml`:
   ```yaml
   command:
     - --quantize=bitsandbytes-nf4
   ```
3. **Reduce batch size**:
   ```yaml
   command:
     - --max-batch-size=4
   ```

### "I don't have enough information..."

This is expected behavior! It means:
- No relevant chunks found in the database
- Your query may need rephrasing
- Data may need to be re-ingested

### Re-running Data Ingestion

If you change data sources in `.env`:

```bash
# Stop and remove the ingestion container
docker-compose rm -s -v ingest

# Rebuild and run ingestion
docker-compose up -d --build ingest

# Monitor progress
docker-compose logs -f ingest
```

### GPU Not Detected

If TGI can't find your GPU:

```bash
# Verify Docker can see GPU
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check nvidia-container-toolkit installation
sudo systemctl status nvidia-container-toolkit

# Restart Docker Desktop
```

### Checking Service Health

```bash
# View all service status
docker-compose ps

# Check specific service logs
docker-compose logs -f <service-name>

# Test TGI directly
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"inputs": "Hello, how are you?"}'

# Restart a service
docker-compose restart <service-name>
```

### Clearing and Resetting

To start completely fresh:

```bash
# Stop all services
docker-compose down -v

# Remove all data (WARNING: Deletes vector DB)
rm -rf qdrant_storage/

# Clear TGI cache
rm -rf ~/.cache/huggingface/

# Rebuild from scratch
docker-compose up -d --build
```

## Usage Tips

### Writing Effective Queries

- **Be specific**: "What is our refund policy?" vs "Tell me about refunds"
- **Use keywords**: Include terms likely in your documentation
- **Ask follow-ups**: The system maintains conversation context
- **Reference sources**: Mention specific documents if known

### Understanding Responses

- **Source citations**: Every answer includes document references
- **Confidence levels**: Responses are only given when context is found
- **No hallucinations**: The system won't make up information
- **Streaming**: Responses stream in real-time for faster feedback

## Performance Optimization

### For Better Speed
- Use smaller models (Mistral-7B vs Mistral-22B)
- Enable quantization (`--quantize=bitsandbytes-nf4`)
- Reduce `top_k` to retrieve fewer chunks
- Increase batch size for concurrent requests

### For Better Quality
- Use larger models (Mistral-22B, Llama-2-13B)
- Increase `top_k` for more context
- Lower `score_threshold` for broader matches
- Use full precision (disable quantization)

### GPU Utilization

Monitor and optimize GPU usage:

```bash
# Check GPU utilization
nvidia-smi dmon

# TGI performance metrics
curl http://localhost:8080/metrics

# Adjust concurrent requests
# In docker-compose.yml:
command:
  - --max-concurrent-requests=256  # Increase for higher throughput
```

## Advanced Configuration

### Multi-GPU Setup

To use multiple GPUs, modify `docker-compose.yml`:

```yaml
tgi:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ['0', '1']  # Use GPU 0 and 1
            capabilities: [gpu]
  command:
    - --num-shard=2  # Shard model across 2 GPUs
```

### Custom Models

To use your own fine-tuned model:

1. Place model in a local directory
2. Mount it in `docker-compose.yml`:
   ```yaml
   volumes:
     - ./models/my-model:/models/my-model
   command:
     - --model-id=/models/my-model
   ```

### Adjusting Retrieval Settings

In `src/main_api.py`, modify:
- `top_k`: Number of chunks retrieved (default: 4)
- `score_threshold`: Minimum similarity score (default: 0.65)
- `chunk_size`: Token size of chunks (default: 400)

### Custom Prompts

Edit the system prompt in `src/main_api.py` to adjust response style, tone, or formatting.

## Security Notes

- All data stays on your machine - nothing sent to external APIs
- Confluence credentials stored in `.env` (add to `.gitignore`)
- Default Grafana password should be changed in production
- API has no authentication by default (add if needed)
- TGI runs locally with no external model downloads after first setup

## Hugging Face Authentication

Some models require authentication. To use gated models:

1. Create a Hugging Face account
2. Generate an access token: https://huggingface.co/settings/tokens
3. Add to `.env`:
   ```bash
   HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxx
   ```

## System Requirements Summary

### Minimum Specs
- NVIDIA GPU: GTX 1060 6GB or better
- System RAM: 16GB
- Disk Space: 20GB free
- Docker: 20.10.0+
- NVIDIA Driver: 525.60.13+

### Recommended Specs
- NVIDIA GPU: RTX 3080 10GB or better
- System RAM: 32GB
- Disk Space: 50GB free (SSD preferred)
- Docker: Latest version
- NVIDIA Driver: Latest stable

## Support

For detailed system architecture and data flow diagrams, see `ARCHITECTURE.md`.

For issues or questions:
1. Check service logs: `docker-compose logs -f <service>`
2. Review Grafana dashboards for errors
3. Verify `.env` configuration
4. Check GPU availability: `nvidia-smi`
5. Review TGI documentation: https://huggingface.co/docs/text-generation-inference

## License

[Your License Here]

## Contributors

[Your Team Here]
