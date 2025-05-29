import os
import itertools
import yaml



def get_params():
  note_emb_dim = [768]   # 4096, 768
  note_form = ['']                 #['', '_impute', '_stack'] 
  embedding_model = ['clinical_bert']
  target_data = ["mimic3"]               
  algorithm = ["ClinicalBert_CQL_Cross_Context_Attention"]                                      
                                          # [BCQ, CQL]     $IQL                                            
                                          # [ClinicalBert]
                                          # [ClinicalBert_CQL_Cross_Context_Attention]
  bcq_threshold = [0.3] # [0.2, 0.3]                       
  discount = [0.98] # 0.99, 0.98                           
  optimizer = ['Adam']
  optimizer_parameters = [{"lr":0.0001, 'weight_decay':0.01}] # {"lr":0.0003}, {"lr":0.001}        
  use_polyak_target_update = [False] # [False, True] 
  target_update_frequency = [10] # [10, 50,100]
  tau = [0.005] # [0.005, 0.1]
  max_timesteps = [1500]
  eval_freq = [10] 
  tol = [2]
  rho_clip = [1]
  hidden_node = [512] # 512 or 1024
  activation = ["relu"] # ["relu", "tanh"]
  batch_size = [256]
  

 
  param_array = list(itertools.product(bcq_threshold, 
                                      discount, 
                                      optimizer, 
                                      optimizer_parameters, 
                                      use_polyak_target_update, 
                                      target_update_frequency,
                                      tau,
                                      algorithm,
                                      max_timesteps,
                                      eval_freq,
                                      tol,
                                      rho_clip,
                                      hidden_node,
                                      activation,
                                      target_data,
                                      batch_size,
                                      embedding_model,
                                      note_form,
                                      note_emb_dim
                                      ))
  param_options = []
  for i in param_array:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    params = yaml.safe_load(open(os.path.join(dir_path, '../configs/config_base.yaml'), 'r'))  #dict type

    params["bcq_threshold"] = i[0]
    params["discount"] = i[1]
    params["optimizer"] = i[2]
    params["optimizer_parameters"] = i[3]
    params["use_polyak_target_update"] = i[4]
    params["target_update_frequency"] = i[5]
    params["tau"] = i[6]
    params["algorithm"] = i[7]
    params["max_timesteps"] = i[8]
    params["eval_freq"] = i[9]
    params["tol"] = i[10]
    params["rho_clip"] = i[11]
    params["hidden_node"] = i[12]
    params["activation"] = i[13]
    params["target_data"] = i[14]
    params["batch_size"] = i[15]
    params["embedding_model"] = i[16]
    params['note_form'] = i[17]
    params['note_emb_dim'] = i[18]
    param_options.append(params)
  return param_options[0]


