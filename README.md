![IMG_098555555555](https://github.com/user-attachments/assets/59d316cc-f006-4f44-841c-cc5b84ad088b)

# Disclaimer...Reading!!!
**`This campaign rewards users who run worker nodes providing inferences for the US presidential election party winner once a day. Every inference should be the likelihood of the republican party winning the election. source`** [run-inference-political](https://app.allora.network/points/campaign/run-inference-political)

## 1. Components
- **Worker**: The node that publishes inferences to the Allora chain.
- **Inference**: A container that conducts inferences, maintains the model state, and responds to internal inference requests via a Flask application. This node operates with a basic linear regression model for price predictions.
- **Updater**: A cron-like container designed to update the inference node's data by daily fetching the latest market information from the data provider, ensuring the model stays current with new market trends.
- **Topic ID**: Running this worker on `TopicId 11`
- **TOKEN= D** For have inference `D`: Democrat
- **TOKEN= R** For have inference `R`: Republic
- **MODEL**: Own your model or modify `model.py`
- **Probability**: Predict of `%` total `0 - 100%`
- **Dataset**: polymarket.com
- **An expected result**: Every `24` hours 

### Setup Worker

1. **Clone this repository**
   ```sh
   git clone https://github.com/arcxteam/allora-usa-election.git
   cd allora-usa-election
    ```
2. **Provided and config environment file modify model-tunning**
    
    Copy and read the example .env.example for your variables
    ```sh
    nano .env.example .env
    ```
    Here are the currently accepted configuration
   - TOKEN= (`D` or `R`)
   - MODEL= the name as model (defaults `SVR` or modify your own model)
   - Save `ctrl X + Y and Enter`
   
   Modify model-tunning or check /models folder
   ```sh
    nano model.py
    ```

4. **Edit your config & initialize worker**

   Edit for key wallet - name wallet - RPC endpoint & interval etc
    ```sh
    nano config.json
    ```
   Run the following commands root directory to initialize the worker
    ```sh
    chmod +x init.config
    ./init.config
    ```
5. **Start the Services**
    
    Run the following command to start the worker node, inference, and updater nodes:
    ```sh
    docker compose up --build -d
    ```
    Check running
    ```sh
    docker compose logs -f --tail=100
    ```

   To confirm that the worker successfully sends the inferences to the chain, look for the following logs:
    ```
    {"level":"debug","msg":"Send Worker Data to chain","txHash":<tx-hash>,"time":<timestamp>,"message":"Success"}
    ```

   ![Capture333654](https://github.com/user-attachments/assets/a61b3779-e80e-4e8a-8518-672f55e30f06)

## 2. Testing Inference Only

   Send requests to the inference model. For example, request probability of Democrat(`D`) or Republic(`R`)
   ```sh
   curl http://127.0.0.1:8000/inference/D
   ```
   Expected response of numbering:
   `
   "value":"xx.xxxx"`
