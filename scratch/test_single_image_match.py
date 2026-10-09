import yaml
import torch
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate import build_model
from src.app.inference_engine import MultimodalInferenceEngine

def main():
    config = yaml.safe_load(open('configs/config.yaml'))
    ckpt = 'outputs/experiments/graph_transformer/checkpoints/best_model.pth'

    m_eval = build_model('graph_transformer', config)
    m_eval.load_state_dict(torch.load(ckpt, map_location='cpu')['model_state_dict'])
    m_eval.eval()

    app_engine = MultimodalInferenceEngine(config_path='configs/config.yaml', gt_checkpoint_path=ckpt, device='cpu')

    img_path_m = 'BCMID/0072y16/Mammogram/0072y16_L_CC_1.png'
    img_path_u = 'BCMID/0072y16/Ultrasound/0072y16_US_L_1.png'

    with open(img_path_m, 'rb') as fm, open(img_path_u, 'rb') as fu:
        m_bytes = fm.read()
        u_bytes = fu.read()

    res_app = app_engine.predict(mammo_input=m_bytes, us_input=u_bytes)

    tm = app_engine.preprocess_image(m_bytes, 'mammogram')
    tu = app_engine.preprocess_image(u_bytes, 'ultrasound')
    idx = torch.zeros(1, dtype=torch.long)

    with torch.no_grad():
        logits, _ = m_eval(tm, idx, tu, idx, batch_size=1)
        eval_prob = F.softmax(logits, dim=1)[0, 1].item()

    print('--- GRAPH TRANSFORMER ---')
    print('App Prob:    ', res_app['prediction']['prob_malignant'])
    print('Eval Prob:   ', eval_prob)
    print('Difference:  ', abs(res_app['prediction']['prob_malignant'] - eval_prob))

    # DenseNet-121
    m_dense_eval = build_model('densenet', config)
    ckpt_d = 'outputs/experiments/densenet_mammogram/checkpoints/best_model.pth'
    m_dense_eval.load_state_dict(torch.load(ckpt_d, map_location='cpu')['model_state_dict'])
    m_dense_eval.eval()

    res_app_d = app_engine.predict(mammo_input=m_bytes)
    with torch.no_grad():
        logits_d, _ = m_dense_eval(tm, idx, batch_size=1)
        eval_prob_d = F.softmax(logits_d, dim=1)[0, 1].item()

    print('\n--- DENSENET-121 (MAMMOGRAM) ---')
    print('App Prob:    ', res_app_d['prediction']['prob_malignant'])
    print('Eval Prob:   ', eval_prob_d)
    print('Difference:  ', abs(res_app_d['prediction']['prob_malignant'] - eval_prob_d))

    # ResNet-50
    m_res_eval = build_model('resnet', config)
    ckpt_r = 'outputs/experiments/resnet_ultrasound/checkpoints/best_model.pth'
    m_res_eval.load_state_dict(torch.load(ckpt_r, map_location='cpu')['model_state_dict'])
    m_res_eval.eval()

    res_app_r = app_engine.predict(us_input=u_bytes)
    with torch.no_grad():
        logits_r, _ = m_res_eval(tu, idx, batch_size=1)
        eval_prob_r = F.softmax(logits_r, dim=1)[0, 1].item()

    print('\n--- RESNET-50 (ULTRASOUND) ---')
    print('App Prob:    ', res_app_r['prediction']['prob_malignant'])
    print('Eval Prob:   ', eval_prob_r)
    print('Difference:  ', abs(res_app_r['prediction']['prob_malignant'] - eval_prob_r))

if __name__ == '__main__':
    main()
