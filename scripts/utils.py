import wandb
from pathlib import Path 
from typing import Optional, Dict 

# Remember to `wandb login` so a wandb password is saved to your ~/.netrc file
def log_results(args, results):
    cfg = {}
    d = args.__dict__
    for k in d:
        cfg[k] = d[k]
    wandb.init(project='probing_dataset', entity="ziningzhu", config=cfg)
    wandb.log(results)


def init_or_resume_wandb_run(wandb_id_file_path: Optional[str] = None,
                             project_name: Optional[str] = None,
                             run_name: Optional[str] = None,
                             config: Optional[Dict] = None):
    """Detect the run id if it exists and resume
        from there, otherwise write the run id to file. 
        
        Returns the config, if it's not None it will also update it first
        
        NOTE:
            Make sure that wandb_id_file_path.parent exists before calling this function
    """
    if wandb_id_file_path is None:
        wandb.init(project=project_name, name=run_name, config=config)
    else:
        # if the run_id was previously saved, resume from there
        pathobj = Path(wandb_id_file_path)
        if pathobj.exists():
            resume_id = pathobj.read_text()
            wandb.init(project=project_name,
                    name=run_name,
                    resume=resume_id,
                    config=config)
        else:
            # if the run_id doesn't exist, then create a new run
            # and write the run id the file
            run = wandb.init(project=project_name, name=run_name, config=config)
            pathobj.write_text(str(run.id))

    wandb_config = wandb.config
    if config is not None:
        # update the current passed in config with the wandb_config
        config.update(wandb_config)

    return config


def find_hf_loc(ud_tokens, hf_tokens, ud_loc=0):
    """
    Inputs:
        ud_tokens: List[str]. One word per token. Assume no words contain punctuations.
        hf_tokens: List[str]. Sometimes tokens start with "##".
        ud_loc: int. Index at ud_tokens
    Outputs:
        hf_loc_start, hf_loc_end: int. Indices at hf_tokens. 
    """
    i, j = 0, 0
    hf_loc_start, hf_loc_end = None, None 
    copy_hf_tokens = hf_tokens[:]
    while i < len(hf_tokens):
        if j == ud_loc:
            if hf_loc_start is None:
                hf_loc_start = i
            else:
                hf_loc_end = i
        i += 1
        if (i < len(hf_tokens)) and (not hf_tokens[i].startswith("##")):
            j += 1
    # Process this, so that I can directly invoke [hf_loc_start:hf_loc_end]
    if hf_loc_end is None:
        hfle = hf_loc_start + 1
    else:
        hfle = hf_loc_end + 1
    return hf_loc_start, hfle
