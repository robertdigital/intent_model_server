# Intent Model Server

WIP

## How to use

### Docker image

```shell
docker pull nvaitc/intent_model_server:latest
nvidia-docker run -p 5000:5000 nvaitc/intent_model_server:latest
```

### Development/source

1. Download model weights from [releases section](https://github.com/NVAITC/intent_model_server/releases).
2. `git clone https://github.com/NVAITC/intent_model_server`
3. Place weights in root folder (`intent_model_server`)
4. `python3 app.py`
5. Server will be running on port `5000` by default


