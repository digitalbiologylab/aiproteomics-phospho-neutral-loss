from peptdeep.pretrained_models import ModelManager
from peptdeep.model.ms2 import pDeepModel, ModelMS2Bert

class CustomModelManager(ModelManager):
    def __init__(self, mask_modloss = False, device = "gpu", ms2_model=pDeepModel):
        super().__init__(mask_modloss, device)
        self.ms2_model = ms2_model

class CustompDeepModel(pDeepModel):
    def _get_modloss_frags(self, modloss=["modloss", 'NH3', 'H2O']):
        self._modloss_frag_types = []
        for i, frag in enumerate(self.charged_frag_types):
            for loss_type in modloss:
                if loss_type in frag:
                    self._modloss_frag_types.append(i)
                    break