# CoT-Guard

## Environment variables

From the project root, set the data and temporary directories and add the project to `PYTHONPATH`:

```bash
export COT_GLOBAL_DIRECTORY="/path/to/data"
export COT_TEMP_DIRECTORY="/path/to/temp"
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_API_KEY="your-gemini-api-key"
export PYTHONPATH=$PYTHONPATH:$(pwd)
```
