import mlflow
from configs.config import get_params
from buffer import ReplayBuffer
from agent import MultimodalContextCQL, TextCQL, TabularCQL
from metric import eval_multi_step_doubly_robust_ci, eval_wis_ci, eval_fqe_ci, eval_policy_survival_rate, eval_opera_ci, collect_bellman_residuals
from util import set_seed
import numpy as np
import torch
import os

def _sanitize(s: str) -> str:
    return str(s).replace("%", "pct").replace("/", "_").replace(" ", "")

def save_checkpoint(policy, params, train_f, training_iters, save_dir="./pth"):
    os.makedirs(save_dir, exist_ok=True)

    fname = (
        f"{_sanitize(params['target_data'])}_"
        f"{_sanitize(params['embedding_model'])}_"
        f"{_sanitize(params['algorithm'])}_"
        f"{_sanitize(params.get('note_form',''))}_"
        f"train{_sanitize(train_f)}_"
        f"seed{_sanitize(params.get('seed',0))}.pt"
    )
    save_path = os.path.join(save_dir, fname)

    params_to_save = dict(params)
    if "device" in params_to_save:
        params_to_save["device"] = str(params_to_save["device"])

    ckpt = {
        "params": params_to_save,
        "train_frac": train_f,
        "training_iters": training_iters,
        "policy_state": {
            "Q": policy.Q.state_dict() if hasattr(policy, "Q") else None,
            "Q_target": policy.Q_target.state_dict() if hasattr(policy, "Q_target") else None,
            "Q_optimizer": policy.Q_optimizer.state_dict() if hasattr(policy, "Q_optimizer") else None,
            "optimizer": policy.optimizer.state_dict() if hasattr(policy, "optimizer") else None,
            "lr_scheduler": policy.lr_scheduler.state_dict() if hasattr(policy, "lr_scheduler") else None,
            "iterations": getattr(policy, "iterations", None),
            "threshold": getattr(policy, "threshold", None),
        }
    }

    torch.save(ckpt, save_path)
    print(f"[Checkpoint Saved] {save_path}")

    return save_path

def log_params(params):

    mlflow.log_param("bcq_threshold", params["bcq_threshold"])
    mlflow.log_param("discount", params["discount"])
    mlflow.log_param("optimizer", params["optimizer"])
    mlflow.log_param("lr", params["optimizer_parameters"]["lr"])
    mlflow.log_param("optimizer_parameters", params["optimizer_parameters"])
    mlflow.log_param("use_polyak_target_update", params["use_polyak_target_update"])
    mlflow.log_param("target_update_frequency", params["target_update_frequency"])
    mlflow.log_param("tau", params["tau"])
    mlflow.log_param("algorithm", params["algorithm"])
    mlflow.log_param("max_timesteps", params["max_timesteps"])
    mlflow.log_param("eval_freq", params["eval_freq"])
    mlflow.log_param("tol", params["tol"])
    mlflow.log_param("rho_clip", params["rho_clip"])
    mlflow.log_param("hidden_node", params["hidden_node"])
    mlflow.log_param("activation", params["activation"])
    mlflow.log_param("seed", params["seed"])
    mlflow.log_param("batch_size", params["batch_size"])    


        
def train(params):

    train_buffer = ReplayBuffer(state_dim=params['state_dim'],
                                embed_dim= params['note_emb_dim'],
                                batch_size=params['batch_size'],
                                target_data=params['target_data'],
                                buffer_path=f"./dataset/{params['target_data']}/buffer_{params['embedding_model']}/",
                                note_form=params['note_form']
                                ).load_original_dataset(only_test_set=False)
    
    test_buffer = ReplayBuffer(state_dim=params['state_dim'],
                                embed_dim= params['note_emb_dim'],
                                batch_size=params['batch_size'],
                                target_data=params['target_data'],
                                buffer_path=f"./dataset/{params['target_data']}/buffer_{params['embedding_model']}/",
                                note_form=params['note_form']
                                ).load_original_dataset(only_test_set=True)
    ###########################################################################################
    val_datasets = ['mimic3', 'mimic4', 'pd']
    val_datasets.remove(params['target_data'])

    val_buffer1 = ReplayBuffer(state_dim=params['state_dim'],
                               embed_dim= params['note_emb_dim'],
                               batch_size=params['batch_size'],
                               target_data=val_datasets[0],
                               buffer_path=f"./dataset/{val_datasets[0]}/buffer_{params['embedding_model']}/",
                               note_form=params['note_form']
                               ).load_validation_dataset(ori_data = params['target_data'])
    
    val_buffer2 = ReplayBuffer(state_dim=params['state_dim'],
                               embed_dim= params['note_emb_dim'],
                               batch_size=params['batch_size'],
                               target_data=val_datasets[1],
                               buffer_path=f"./dataset/{val_datasets[1]}/buffer_{params['embedding_model']}/",
                               note_form=params['note_form']
                               ).load_validation_dataset(ori_data = params['target_data'])

    ################################################################################
    policy = MultimodalContextCQL(**params)
    # policy = TextCQL(**params)
    # policy = TabularCQL(**params)
    ################################################################################

    training_iters = 0
    while training_iters < params["max_timesteps"]: 
        policy.train(train_buffer)
        if training_iters % params["eval_freq"] == 0:
            print(f"start eval: {training_iters}")

            ######################################################################################################################
            
            
        training_iters += 1
        # print(training_iters)
    # save_checkpoint(policy, params, train_f=train_f, training_iters=training_iters, save_dir="./pth")
    # bellman_residuals = collect_bellman_residuals(params['algorithm'], policy, test_buffer)
    # np.save(f"./residuals_{params['algorithm']}", bellman_residuals)

    ######################################################################################################################
    dr, dr_low, dr_high= eval_multi_step_doubly_robust_ci(params['algorithm'], policy, test_buffer, params['rho_clip'], params['batch_size'], params['discount'])
    mlflow.log_metric(f"{params['target_data']} dr", dr, step=training_iters)
    mlflow.log_metric(f"{params['target_data']} dr ci low", dr_low, step=training_iters)
    mlflow.log_metric(f"{params['target_data']} dr ci high", dr_high, step=training_iters)
    
    dr, dr_low, dr_high = eval_multi_step_doubly_robust_ci(params['algorithm'], policy, val_buffer1, params['rho_clip'], params['batch_size'], params['discount'])
    mlflow.log_metric(f"{val_datasets[0]} dr", dr, step=training_iters)
    mlflow.log_metric(f"{val_datasets[0]} dr ci low", dr_low, step=training_iters)
    mlflow.log_metric(f"{val_datasets[0]} dr ci high", dr_high, step=training_iters)

    dr, dr_low, dr_high = eval_multi_step_doubly_robust_ci(params['algorithm'], policy, val_buffer2, params['rho_clip'], params['batch_size'], params['discount'])
    mlflow.log_metric(f"{val_datasets[1]} dr", dr, step=training_iters)
    mlflow.log_metric(f"{val_datasets[1]} dr ci low", dr_low, step=training_iters)
    mlflow.log_metric(f"{val_datasets[1]} dr ci high", dr_high, step=training_iters)
    ######################################################################################################################
    fqe, fqe_low, fqe_high = eval_fqe_ci(params['algorithm'], policy, test_buffer, params['discount'])
    mlflow.log_metric(f"{params['target_data']} fqe", fqe, step=training_iters)
    mlflow.log_metric(f"{params['target_data']} fqe ci low", fqe_low, step=training_iters)
    mlflow.log_metric(f"{params['target_data']} fqe ci high", fqe_high, step=training_iters)

    fqe, fqe_low, fqe_high = eval_fqe_ci(params['algorithm'], policy, val_buffer1, params['discount'])
    mlflow.log_metric(f"{val_datasets[0]} fqe", fqe, step=training_iters)
    mlflow.log_metric(f"{val_datasets[0]} fqe ci low", fqe_low, step=training_iters)
    mlflow.log_metric(f"{val_datasets[0]} fqe ci high", fqe_high, step=training_iters)

    fqe, fqe_low, fqe_high = eval_fqe_ci(params['algorithm'], policy, val_buffer2, params['discount'])
    mlflow.log_metric(f"{val_datasets[1]} fqe", fqe, step=training_iters)
    mlflow.log_metric(f"{val_datasets[1]} fqe ci low", fqe_low, step=training_iters)
    mlflow.log_metric(f"{val_datasets[1]} fqe ci high", fqe_high, step=training_iters)
    ######################################################################################################################
    wis, wis_low, wis_high = eval_wis_ci(params['algorithm'], policy, test_buffer, params['rho_clip'], params['discount'])
    mlflow.log_metric(f"{params['target_data']} wis", wis, step=training_iters)
    mlflow.log_metric(f"{params['target_data']} wis ci low", wis_low, step=training_iters)
    mlflow.log_metric(f"{params['target_data']} wis ci high", wis_high, step=training_iters)

    wis, wis_low, wis_high = eval_wis_ci(params['algorithm'], policy, val_buffer1, params['rho_clip'], params['discount'])
    mlflow.log_metric(f"{val_datasets[0]} wis", wis, step=training_iters)
    mlflow.log_metric(f"{val_datasets[0]} wis ci low", wis_low, step=training_iters)
    mlflow.log_metric(f"{val_datasets[0]} wis ci high", wis_high, step=training_iters)

    wis, wis_low, wis_high = eval_wis_ci(params['algorithm'], policy, val_buffer2, params['rho_clip'], params['discount'])
    mlflow.log_metric(f"{val_datasets[1]} wis", wis, step=training_iters)
    mlflow.log_metric(f"{val_datasets[1]} wis ci low", wis_low, step=training_iters)
    mlflow.log_metric(f"{val_datasets[1]} wis ci high", wis_high, step=training_iters)
    ######################################################################################################################
    opera, opera_low, opera_high, _ = eval_opera_ci(params['algorithm'], policy, test_buffer)
    mlflow.log_metric(f"{params['target_data']} opera", opera, step=training_iters)
    mlflow.log_metric(f"{params['target_data']} opera ci low", opera_low, step=training_iters)
    mlflow.log_metric(f"{params['target_data']} opera ci high", opera_high, step=training_iters)

    opera, opera_low, opera_high, _ = eval_opera_ci(params['algorithm'], policy, val_buffer1)
    mlflow.log_metric(f"{val_datasets[0]} opera", opera, step=training_iters)
    mlflow.log_metric(f"{val_datasets[0]} opera ci low", opera_low, step=training_iters)
    mlflow.log_metric(f"{val_datasets[0]} opera ci high", opera_high, step=training_iters)

    opera, opera_low, opera_high, _ = eval_opera_ci(params['algorithm'], policy, val_buffer2)
    mlflow.log_metric(f"{val_datasets[1]} opera", opera, step=training_iters)
    mlflow.log_metric(f"{val_datasets[1]} opera ci low", opera_low, step=training_iters)
    mlflow.log_metric(f"{val_datasets[1]} opera ci high", opera_high, step=training_iters)
    ######################################################################################################################
    tot_sr_low, tot_sr_low_ci, tot_sr_high, tot_sr_high_ci, vaso_sr_low, vaso_sr_low_ci, vaso_sr_high, vaso_sr_high_ci, iv_sr_low, iv_sr_low_ci, iv_sr_high, iv_sr_high_ci = eval_policy_survival_rate(params['algorithm'], policy, test_buffer, params['tol'])
    mlflow.log_metric(f"{params['target_data']} total sr similar", tot_sr_low, step=training_iters)
    mlflow.log_metric(f"{params['target_data']} total sr similar low", tot_sr_low_ci[0], step=training_iters)
    mlflow.log_metric(f"{params['target_data']} total sr similar high", tot_sr_low_ci[1], step=training_iters)

    mlflow.log_metric(f"{params['target_data']} total sr no similar", tot_sr_high, step=training_iters)
    mlflow.log_metric(f"{params['target_data']} total sr no similar low", tot_sr_high_ci[0], step=training_iters)
    mlflow.log_metric(f"{params['target_data']} total sr no similar high", tot_sr_high_ci[1], step=training_iters)

    mlflow.log_metric(f"{params['target_data']} vaso_sr similar", vaso_sr_low, step=training_iters)
    mlflow.log_metric(f"{params['target_data']} vaso_sr similar low", vaso_sr_low_ci[0], step=training_iters)
    mlflow.log_metric(f"{params['target_data']} vaso_sr similar high", vaso_sr_low_ci[1], step=training_iters)

    mlflow.log_metric(f"{params['target_data']} vaso_sr no similar", vaso_sr_high, step=training_iters)
    mlflow.log_metric(f"{params['target_data']} vaso_sr no similar low", vaso_sr_high_ci[0], step=training_iters)
    mlflow.log_metric(f"{params['target_data']} vaso_sr no similar high", vaso_sr_high_ci[1], step=training_iters)

    mlflow.log_metric(f"{params['target_data']} iv_sr similar", iv_sr_low, step=training_iters)
    mlflow.log_metric(f"{params['target_data']} iv_sr similar low", iv_sr_low_ci[0], step=training_iters)
    mlflow.log_metric(f"{params['target_data']} iv_sr similar high", iv_sr_low_ci[1], step=training_iters)

    mlflow.log_metric(f"{params['target_data']} iv_sr no similar", iv_sr_high, step=training_iters)
    mlflow.log_metric(f"{params['target_data']} iv_sr no similar low", iv_sr_high_ci[0], step=training_iters)
    mlflow.log_metric(f"{params['target_data']} iv_sr no similar high", iv_sr_high_ci[1], step=training_iters)

    tot_sr_low, tot_sr_low_ci, tot_sr_high, tot_sr_high_ci, vaso_sr_low, vaso_sr_low_ci, vaso_sr_high, vaso_sr_high_ci, iv_sr_low, iv_sr_low_ci, iv_sr_high, iv_sr_high_ci = eval_policy_survival_rate(params['algorithm'], policy, val_buffer1, params['tol'])
    mlflow.log_metric(f"{val_datasets[0]} total sr similar", tot_sr_low, step=training_iters)
    mlflow.log_metric(f"{val_datasets[0]} total sr similar low", tot_sr_low_ci[0], step=training_iters)
    mlflow.log_metric(f"{val_datasets[0]} total sr similar high", tot_sr_low_ci[1], step=training_iters)

    mlflow.log_metric(f"{val_datasets[0]} total sr no similar", tot_sr_high, step=training_iters)
    mlflow.log_metric(f"{val_datasets[0]} total sr no similar low", tot_sr_high_ci[0], step=training_iters)
    mlflow.log_metric(f"{val_datasets[0]} total sr no similar high", tot_sr_high_ci[1], step=training_iters)

    mlflow.log_metric(f"{val_datasets[0]} vaso_sr similar", vaso_sr_low, step=training_iters)
    mlflow.log_metric(f"{val_datasets[0]} vaso_sr similar low", vaso_sr_low_ci[0], step=training_iters)
    mlflow.log_metric(f"{val_datasets[0]} vaso_sr similar high", vaso_sr_low_ci[1], step=training_iters)

    mlflow.log_metric(f"{val_datasets[0]} vaso_sr no similar", vaso_sr_high, step=training_iters)
    mlflow.log_metric(f"{val_datasets[0]} vaso_sr no similar low", vaso_sr_high_ci[0], step=training_iters)
    mlflow.log_metric(f"{val_datasets[0]} vaso_sr no similar high", vaso_sr_high_ci[1], step=training_iters)

    mlflow.log_metric(f"{val_datasets[0]} iv_sr similar", iv_sr_low, step=training_iters)
    mlflow.log_metric(f"{val_datasets[0]} iv_sr similar low", iv_sr_low_ci[0], step=training_iters)
    mlflow.log_metric(f"{val_datasets[0]} iv_sr similar high", iv_sr_low_ci[1], step=training_iters)

    mlflow.log_metric(f"{val_datasets[0]} iv_sr no similar", iv_sr_high, step=training_iters)
    mlflow.log_metric(f"{val_datasets[0]} iv_sr no similar low", iv_sr_high_ci[0], step=training_iters)
    mlflow.log_metric(f"{val_datasets[0]} iv_sr no similar high", iv_sr_high_ci[1], step=training_iters)

    tot_sr_low, tot_sr_low_ci, tot_sr_high, tot_sr_high_ci, vaso_sr_low, vaso_sr_low_ci, vaso_sr_high, vaso_sr_high_ci, iv_sr_low, iv_sr_low_ci, iv_sr_high, iv_sr_high_ci = eval_policy_survival_rate(params['algorithm'], policy, val_buffer2, params['tol'])
    mlflow.log_metric(f"{val_datasets[1]} total sr similar", tot_sr_low, step=training_iters)
    mlflow.log_metric(f"{val_datasets[1]} total sr similar low", tot_sr_low_ci[0], step=training_iters)
    mlflow.log_metric(f"{val_datasets[1]} total sr similar high", tot_sr_low_ci[1], step=training_iters)

    mlflow.log_metric(f"{val_datasets[1]} total sr no similar", tot_sr_high, step=training_iters)
    mlflow.log_metric(f"{val_datasets[1]} total sr no similar low", tot_sr_high_ci[0], step=training_iters)
    mlflow.log_metric(f"{val_datasets[1]} total sr no similar high", tot_sr_high_ci[1], step=training_iters)

    mlflow.log_metric(f"{val_datasets[1]} vaso_sr similar", vaso_sr_low, step=training_iters)
    mlflow.log_metric(f"{val_datasets[1]} vaso_sr similar low", vaso_sr_low_ci[0], step=training_iters)
    mlflow.log_metric(f"{val_datasets[1]} vaso_sr similar high", vaso_sr_low_ci[1], step=training_iters)

    mlflow.log_metric(f"{val_datasets[1]} vaso_sr no similar", vaso_sr_high, step=training_iters)
    mlflow.log_metric(f"{val_datasets[1]} vaso_sr no similar low", vaso_sr_high_ci[0], step=training_iters)
    mlflow.log_metric(f"{val_datasets[1]} vaso_sr no similar high", vaso_sr_high_ci[1], step=training_iters)

    mlflow.log_metric(f"{val_datasets[1]} iv_sr similar", iv_sr_low, step=training_iters)
    mlflow.log_metric(f"{val_datasets[1]} iv_sr similar low", iv_sr_low_ci[0], step=training_iters)
    mlflow.log_metric(f"{val_datasets[1]} iv_sr similar high", iv_sr_low_ci[1], step=training_iters)

    mlflow.log_metric(f"{val_datasets[1]} iv_sr no similar", iv_sr_high, step=training_iters)
    mlflow.log_metric(f"{val_datasets[1]} iv_sr no similar low", iv_sr_high_ci[0], step=training_iters)
    mlflow.log_metric(f"{val_datasets[1]} iv_sr no similar high", iv_sr_high_ci[1], step=training_iters)
    #####################################################################################################################
    
        

    
if __name__ == "__main__":
    params = get_params()
    # set_seed(params['seed'])

    mlflow.set_experiment(params['target_data'])
    with mlflow.start_run(run_name=f"{params['algorithm']}_{params['seed']}"):
        log_params(params)
        train(params)
