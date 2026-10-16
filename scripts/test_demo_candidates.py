import sys, os
sys.path.insert(0, os.path.abspath("."))
from predict_sample import predict_sample

imgs = {
    "low_risk_mam": ("BCMID/03913y21/Mammogram/03913y21_L_CC_2.png", ["exp1", "exp3"]),
    "low_risk_us": ("BCMID/03913y21/Ultrasound/03913y21_US_L_1.png", ["exp2", "exp4"]),
    "susp_mam": ("BCMID/00506y21/Mammogram/00506y21_L_CC_2.png", ["exp1", "exp3"]),
    "susp_us": ("BCMID/00506y21/Ultrasound/00506y21_US_L_1.png", ["exp2", "exp4"]),
}

for name, (path, models) in imgs.items():
    print(f"\n=== {name} ({path}) ===")
    for m in models:
        res = predict_sample(path, model_key=m)
        pred = res["predicted_class"]
        p_low = res["probability_low_risk_percent"]
        p_susp = res["probability_suspicious_percent"]
        print(f"  Model {m.upper()} ({res['model_name']}, {res['modality']}) -> Pred: {pred} | P(Low)={p_low} | P(Susp)={p_susp}")
