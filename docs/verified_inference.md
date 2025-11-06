# Verified 🤝 Inference

Trust but verify.

Miners can opt-in and proxy their LLM requests through a verifier, which will forward their request to the defined provider and respond back with a signed response + proof.
The response with proof is submitted to the querying Validator who can independently verify the proof and adjust rewards if applicable.

## Miner Setup

Just add '--verified.inference' to your startup command and your requests will proxy through the verifier

`pm2 start ./neurons/miner.py --name miner1 -- --netuid 296 --subtensor.network wss://test.finney.opentensor.ai:443 --wallet.name default --wallet.hotkey default --logging.trace --llm.model x-ai/grok-4-fast --verified.inference`

## Validator Setup

No changes needed, verified inference is opt-in for miners.

## Endpoints

Verifier docs

https://testnet.verified.bitrecs.ai/docs

Verifier logs

https://testnet.verified.bitrecs.ai/log

Verifier health

https://testnet.verified.bitrecs.ai/health

## Trusted Inference
<img src="verified_inference.png" alt="Verified Inference" style="border: solid 1px #059669; padding: 2" title="Verified Inference"/>


## Known Limitations

- local miners are excluded from this implementation 
- ed25519 keys are ephemeral

## Roadmap

- enable trusted local setups
- enable multiple signers