Instructions on using this Repo:
1. Always create a PR to main and never force push
2. Agent, AI_Agent, Backend are three different sub modules which represent customer VM, our agent , Data Plane respecctively.
3. Get into the individual modules and run this ./script.sh to build the container and start it.
4. Check the logs using docker logs -f {container name}

Additional Instructions:
1. To run ai_agent and talk to a model, currently it has been configured to use mistral-7B to run locally
2. Install ollama: ```brew install ollama```
3. Run this command to install mistral: ```ollama run mistral:latest``` 
4. ollama serve to check if mistral is already running