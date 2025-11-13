from datetime import datetime, timezone
from bitrecs.utils.rarity import RarityReport
from bitrecs.validator.reward import get_rarity_report


def test_can_fetch_rarity_reports():
    reports = RarityReport.get_reports()
    now = datetime.now(timezone.utc)

    assert isinstance(reports, list)
    assert len(reports) > 0
    for report in reports:
        assert hasattr(report, "model")
        assert hasattr(report, "count")
        assert hasattr(report, "rarity")
        assert hasattr(report, "bonus")
        
        created_at = datetime.fromisoformat(report.created_at)
        assert created_at <= now
        assert report.bonus >= 1.0

    
def test_get_model_rarity_bonus():
    responses = [
        {
            "id": "gen-1762174546-KTPAqRYyKnHjfyGjaXs8",
            "provider": "Google AI Studio",
            "model": "google/gemini-2.0-flash-001",
            "object": "chat.completion",
            "created": 1762174546,
            "choices": [
                {
                    "logprobs": None,
                    "finish_reason": "stop",
                    "native_finish_reason": "STOP",
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "```json\n[\n  {\n    \"sku\": \"WS02\",\n    \"name\": \"Gabrielle Micro Sleeve Top - Clothing|New Luma Yoga Collection|Tees\",\n    \"price\": \"28\",\n    \"reason\": \"This top complements the parachute pants as part of the New Luma Yoga Collection\"\n  },\n  {\n    \"sku\": \"WH10\",\n    \"name\": \"Helena Hooded Fleece - Clothing|Hoodies amp Sweatshirts|New Luma Yoga Collection\",\n    \"price\": \"55\",\n    \"reason\": \"This hoodie provides warmth and style and is part of the New Luma Yoga Collection\"\n  },\n  {\n    \"sku\": \"WJ11\",\n    \"name\": \"Neve Studio Dance Jacket - Clothing|Jackets|New Luma Yoga Collection\",\n    \"price\": \"69\",\n    \"reason\": \"This jacket is a stylish layering piece from the New Luma Yoga Collection\"\n  },\n  {\n    \"sku\": \"MP03\",\n    \"name\": \"Geo Insulated Jogging Pant - Clothing|New Luma Yoga Collection|Pants\",\n    \"price\": \"51\",\n    \"reason\": \"These jogging pants offer warmth and comfort and are part of the New Luma Yoga Collection\"\n  },\n  {\n    \"sku\": \"MP05\",\n    \"name\": \"Kratos Gym Pant - Clothing|New Luma Yoga Collection|Pants\",\n    \"price\": \"57\",\n    \"reason\": \"These gym pants are a comfortable and stylish option from the New Luma Yoga Collection\"\n  },\n  {\n    \"sku\": \"MP07\",\n    \"name\": \"Thorpe Track Pant - Clothing|New Luma Yoga Collection|Pants\",\n    \"price\": \"68\",\n    \"reason\": \"These track pants are a versatile option from the New Luma Yoga Collection\"\n  },\n  {\n    \"sku\": \"MP08\",\n    \"name\": \"Zeppelin Yoga Pant - Clothing|New Luma Yoga Collection|Pants\",\n    \"price\": \"82\",\n    \"reason\": \"These yoga pants are a comfortable and stylish option from the New Luma Yoga Collection\"\n  },\n  {\n    \"sku\": \"WSH03\",\n    \"name\": \"Gwen Drawstring Bike Short - Clothing|New Luma Yoga Collection|Performance Fabrics|Shorts\",\n    \"price\": \"50\",\n    \"reason\": \"These bike shorts are a comfortable and stylish option from the New Luma Yoga Collection\"\n  }\n]\n```",
                        "refusal": None
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 6350,
                "completion_tokens": 569,
                "total_tokens": 6919,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0, "image_tokens": 0}
            }
        },
        {
            "id": "chatcmpl-CXoO4WZJeKccWAyTy97AgExo6cBLw",
            "object": "chat.completion",
            "created": 1762174548,
            "model": "gpt-4o-mini-2024-07-18",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I assist you today?",
                        "refusal": None,
                        "annotations": []
                    },
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 9,
                "total_tokens": 17,
                "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0, "audio_tokens": 0, "accepted_prediction_tokens": 0, "rejected_prediction_tokens": 0}
            },
            "service_tier": "default",
            "system_fingerprint": "fp_560af6e559"
        },
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": "```json\n[\n  {\n    \"sku\": \"MP03\",\n    \"name\": \"Geo Insulated Jogging Pant - Clothing|New Luma Yoga Collection|Pants\",\n    \"price\": \"51\",\n    \"reason\": \"These jogging pants complement the shorts as part of the New Luma Yoga Collection providing a full outfit\"\n  },\n  {\n    \"sku\": \"MS09\",\n    \"name\": \"Ryker LumaTechtrade Tee Crew-neck - Tees\",\n    \"price\": \"32\",\n    \"reason\": \"This tee pairs well with the shorts for a complete athletic outfit\"\n  },\n  {\n    \"sku\": \"MT05\",\n    \"name\": \"Rocco Gym Tank - Tanks\",\n    \"price\": \"24\",\n    \"reason\": \"This tank top is a great option for workouts and complements the active shorts\"\n  },\n  {\n    \"sku\": \"24-MG03\",\n    \"name\": \"Summit Watch - New Luma Yoga Collection|Watches\",\n    \"price\": \"54\",\n    \"reason\": \"This watch from the New Luma Yoga Collection complements the shorts and enhances the athletic look\"\n  },\n  {\n    \"sku\": \"24-UG05\",\n    \"name\": \"Go-Getr Pushup Grips - Fitness Equipment|New Luma Yoga Collection\",\n    \"price\": \"19\",\n    \"reason\": \"These pushup grips enhance the workout experience when paired with the active shorts\"\n  },\n  {\n    \"sku\": \"MSH06\",\n    \"name\": \"Lono Yoga Short - Clothing|New Luma Yoga Collection|Shorts\",\n    \"price\": \"32\",\n    \"reason\": \"This yoga short is another option from the New Luma Yoga Collection providing a similar style\"\n  },\n  {\n    \"sku\": \"MP05\",\n    \"name\": \"Kratos Gym Pant - Clothing|New Luma Yoga Collection|Pants\",\n    \"price\": \"57\",\n    \"reason\": \"These gym pants from the New Luma Yoga Collection provide an alternative to shorts for cooler weather\"\n  },\n  {\n    \"sku\": \"24-MB05\",\n    \"name\": \"Wayfarer Messenger Bag - Bags|New Luma Yoga Collection\",\n    \"price\": \"45\",\n    \"reason\": \"This messenger bag from the New Luma Yoga Collection is great for carrying workout gear\"\n  }\n]\n```",
                        "role": "assistant"
                    }
                }
            ],
            "created": 1762174634,
            "id": "paYIac_XLdqr-8YP96e_2Ao",
            "model": "qwen3-next-80b-a3b-instruct",
            "object": "chat.completion",
            "usage": {
                "completion_tokens": 558,
                "prompt_tokens": 6353,
                "total_tokens": 6911
            }
        }
    ]

    normalized_models = normalize_models(responses)
    rarity_reports = RarityReport.get_reports()
    for model in normalized_models:
        match = get_rarity_report(model, rarity_reports)
        if match is None:
            print(f"No rarity report found for model: {model}")
            continue
        
        assert match.bonus >= 1.0   
        print(f"\033[32m Model: {model} | Rarity: {match.rarity} | Bonus: {match.bonus} \033[0m")

    
def normalize_models(responses) -> set:
    models = set()
    for response in responses:
        model_name = response["model"]
        model_name = model_name.split('/')[-1] if '/' in model_name else model_name
        models.add(model_name)
    return models


