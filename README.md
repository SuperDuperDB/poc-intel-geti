# poc-geti
## Step1: 

Add the OPENAI KEY in deploy/users.env

```
OPENAI_API_KEY: "xxxxx"
```


Start docker-compose

```
docker compose -f docker-compose.yaml up -d
```

## Step2:

Open https://localhost:8888


login notebook and run `build.ipynb`


## Step3:

**Streamlit APP**

Open https://localhost:8501


**FastAPI**

Open https://localhost:18000/docs
