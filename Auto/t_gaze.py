import torch
import torch.nn.functional as F
from torchvision import transforms
from m_model import model_static

model_weight = 'models/model_weights.pkl'
test_transforms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# load model weights
model = model_static(model_weight)
model_dict = model.state_dict()
snapshot = torch.load(model_weight)
model_dict.update(snapshot)
model.load_state_dict(model_dict)
model.cuda()
model.train(False)

def get_gaze_estimate(face):    
    img = test_transforms(face)
    img.unsqueeze_(0)

    output = model(img.cuda())
    score = F.sigmoid(output).item()
    return score
