<div align="center">

# Bitrecs Subnet 

<img src="docs/light-logo.svg#gh-light-mode-only" width="400" height="auto" alt="Bitrecs Logo"/>
<img src="docs/dark-logo.svg#gh-dark-mode-only" width="400" height="auto" alt="Bitrecs Logo"/>

[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

[Discord](https://discord.gg/bittensor) • [Website](https://bitrecs.ai/)
</div>


## Introduction
Bitrecs is a novel recommendation engine</a> built on the Bittensor network. This implementation provides a framework for serving realtime e-commerce recommendations using LLMs. Miners are encouraged to experiment with their own implementations to improve latency and quality. We aim to provide relevant and timely recommendations for any ecommerce site.

### RecSys Landscape

<img src="docs/bitrecs_q.png" alt="basic query" style="border: solid 3px #059669;" title="Recommender complexity"/>


### Mining
Want to earn tao by mining? Check out our [mining guide](docs/running_miner.md) to get started.

### Validating 
Interested in running a validator? See our [validator setup instructions](docs/running_validator.md) for details.

## Subnet
- Testnet: 296
- Mainnet: 122

## Protocol

<img src="docs/bitrecs_08.png" alt="basic query" style="border: solid 3px #059669;"/>


## Subnet Goals
 - **Maximize sales for online retailers (ecommerce merchants)**

 - **Discover the best and most consistent LLMs that can produce quality, generalized recs**

Common objectives shared by merchants:

1) Increase add to carts (engagement)
2) Increase the number of customers who complete an order (conversion rate)
3) Increase the average size of cart (average order value)

The subnet miners are incentivized to try to maximize the goals for online merchants using *existing store data and traffic* (onsite). 

## Problem
Global e-commerce is an estimated 20T market with a ~15% CAGR. A large portion of online sales are directly attributed to onsite recommendations (e.g customers who bought x also bought y, bundled products, last minute checkout items etc). Famously, Amazon attributes 35% of their total online sales</a> to their proprietary recommendation engine.  Unfortunately this engine is out of reach for most online retailers who are operating on smaller platforms (e.g Shopify, Woocommerce etc) which have their own recommendation solutions and widgets, often using older rules-based technology.

This problem is a subset of the much larger General Recommendations space which powers most of the Internet today (e.g Netflix movie recommendations, TikTok reels, Google search suggestions, etc)

We are addressing a more narrow but vital area of e-commerce - [onsite product recommendations for merchants](https://www.perplexity.ai/search/onsite-product-recommendations-PmY8e84GSxavnmdSW9934A)

## Solution

Incentivize Bittensor miners to provide realtime product recommendations for online merchants. 

The first generation solution uses prompting and ICL via a Voting Ensemble method to coerce recs from the latest LLMs. We explore dozens of LLMs and measure their "agreement" via sku recommendations they produce in zero-shot queries. These recs make their way back to client websites for the end user (the shopper) to consume. 

<img src="docs/bitrecs_moe.png" alt="basic query" style="border: solid 3px #059669;"/>

0.8 uses a simple prompt template that we hope to evolve into a fully featured recommender as the network matures. 
Additionally, many LLM's have encoded valuable information about shopping cohorts, seasonality, brand affinities, customer journeys etc which we try to unlock using prompting.

The solution has been designed to offer merchants a free and simple plugin that works out of the box, while hiding all the complexity and abstractions of Bittensor away from them so they can continue to focus on selling and running an online business. Subnet miners can focus on competing and evolving the prompting to produce increasingly valuable and timely product recommendations.

Validators provide the gateway to the miners, which send and collect requests through the network, scoring them and selecting a top candidate to return to the client. 

## Product
The subnet operates through an incentive mechanism where miners produce arrays of product SKUs from a given input SKUs, a catalog of store inventory and prompting. The protocol enables:

- Simple integration for e-commerce merchants through our plugins
- Base miner class with support for several popular LLM model providers
- Validator API proxy for marshalling requests between e-commerce sites and the bittensor network

## Incentive
The incentive mechanism ranks miner responses by performance metrics and sales events such as:

- Latency to response
- Diversity of recs
- Quality of responses
- Coherence to prompt parameters
- End user actions such as add to cart and purchases
- Distance comparison to their peer miners

## Opportunity

We think this is a great opportunity to showcase the power of Bittensor incentives with e-commerce and to create a fully aligned network of parties. The barrier to entry for miners is low on this network as a GPU is not required to mine. The consensus alogirthm is based on the idea that very-large LMs, when queried and ensembled together should provide better overall recs if 
they can agree on a set.

Over time, as the baseline gets set (i.e quality recs at a max request time < 3 seconds) the sales focused incentive mechanism drives miners to search and compete for more impactful recommendations. We hope that we can fine tune and balance the incentive mechanism to achieve a network that is always performing a 'best effort' on every request, with outsized rewards given to top performers which drive real, measurable value for merchants and helping them compete with the incumbents.

## Evolution

As the subnet evolves we get closer to **personalized recommendations** which is what Netflix famously prized 1M for in 2006. Back then, the approach was to use matrix factoring to build giant user-item tables but that came at the cost of having very poor coldstart/zero shot peformance. Today, LLM's excel at zero shot recs as acknowledged by Netflix in their <a href="https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39" target="_blank">recent update</a>.

 This network is specialized for ecommerce but can be adopted for a more general scenario.  

Smaller retailers need access to this technology to remain competitive, and our solution is built to work with existing onsite catalogs (we never recommend 3rd party products or divert traffic to other sites - this is critical to winning merchants trust)

Merchants put enormous time and $ into their online stores, constantly updating and polishing them to make them convert higher. These are fantastic websites to showcase Bittensor technology as they generally are very well designed, work across many devices, are fast and are often being updated with new products/services.

We hope to achieve a state where customers happy to get better overall recommendations, merchants win with higher sales and conversions and miners/validators are rewarded for best effort on every request.

## Privacy

Bitrecs does not use, collect or farm any customer data, PII or anything related to the end user which would otherwise be considered invasive or intrusive.  Our solution is privacy preserving, with final control of what data gets sent to the network in the merchants hands via the plugin.  By default, minimal, anonymous information is gathered to generate the recommendations.

## Roadmap

### Q1 2025
- [x] launch testnet supporting Shopify/Woocommerce
- [x] update scoring to use distance to try to measure LLM agreement across sets
- [x] complete woo/shopify extensions

### Q2 2025
- [x] extension marketplace listings
- [x] expand miner diversity
- [x] plugin features (cart history, persona switching)

### Q3 2025
- [x] launch mainnet - 0.8.6

### Q4 2025
- [x] reasoning scoring - 0.8.7
- [x] verified inference - 0.8.8
- [x] world of bitrecs - 0.8.9

### Q1 2026
- [ ] prompt evolution (v2)

## Getting Started

### Mining
Want to earn tao by mining? Check out our [mining guide](docs/running_miner.md) to get started.

### Validating 
Interested in running a validator? See our [validator setup instructions](docs/running_validator.md) for details.

### Testing

``` 
Ensure you have the sample .json and .csv files in the /tests/data folder

pytest ./tests/test_json.py -s --durations=0
pytest ./tests/test_llm.py -s --durations=0
pytest ./tests/test_llm_2.py -s -k 'test_call_local_llm_with_20k'
 ```

### References
<a href="https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39" target="_blank">Netflix Personalized Recommendations</a>

<a href="https://www.perplexity.ai/search/recommendation-engine-NpaNi7MHQ5OFA.btgIQ4QQ" target="_blank">RecSys Perplexity</a>

<a href="https://www.perplexity.ai/search/how-much-does-amazon-product-r-ccxRVV5OReGL_12_QzBGzQ" target="_blank">Amazon Sales</a>

<a href="https://en.wikipedia.org/wiki/Netflix_Prize" target="_blank">Netflix Prize</a>

## License

This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# Copyright © 2024 Bitrecs

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```
