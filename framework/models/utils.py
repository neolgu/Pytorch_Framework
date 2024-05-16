from .model_1 import BaseCNN


def get_model(model_name, **kwargs):
    model_classes = {
        "BaseCNN": BaseCNN,
        # other model...
    }

    # 주어진 모델 이름이 있는지 확인하고 해당 모델 클래스를 반환합니다.
    if model_name in model_classes:
        return model_classes[model_name](**kwargs)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
