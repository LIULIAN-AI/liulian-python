### prompt 2026.02.27 - 1:

Great! I think run.py is now working very well now. However as we can see the code in run.py is still a bit too complicated. We have already done something in cli.py whether we wrapper up the whole procedure so that users can call with simple command with args. Based on that, please do the following:
- Please check current cli.py and run.py, identify all features in run.py, and upgrade cli.py accordingly.
- Backup current run.py, and write a new run.py which uses wrapped up codes in cli. The code in new run.py should be very simple and straightforward. Notice, you have to keep ALL SETTINGS EXACTLY THE SAME as in current run.py. All procedures, mid and final outcomes MUST be exactly the same as current run.py.
- Current run.py have a lot of settings. Please keep them. Meanwhile, implement a config system, where users can set up all settings in a file (e.g., yaml), and cli can choose to read settings from that file.
You may need to research these tasks in parallel first to make proper coding plans. Please research the current code first to determine if there are already some parts implemented. Make explicit plans and details of revision first, and then do the revision. Please follow the coding conventions in /project-adaptor. Please also check online the skill superpowers (https://github.com/obra/superpowers) to see if it is useful for us. If so, you can tell me how you plan to use this skill, download this skill as well and use it as a reference.


### prompt 2026.02.27 - 2:

Great! Now do this: Check if there are already e2e test codes for using swissriver 1990 dataset (a every small part) for the whole run_experiment pipeline, including the following tasks:
- Ray tuning with embedding
- Ray tuning without embedding
- Single run with embedding
- Single run without embedding
If there are already test, revise them to following the run.py procedure; if not, create these tests. Tests should following these rules:
- Not only assert shapes of the outputs, but also the final results (with given seeds), including predicted y and metrics. They should be the same (or very close).
- This test should only be ran if modification is made for the main branch.
- You should however run these test twice to check if it works. The first time, you need to have results and hard recorded them for future re-tests; the second time, you should do the retests to see if the results are the same.
- Given me a report for the test.
After finishing this, give me a list of other tests you see needed. The point is that, we might have heavy revision of the codes to adapt other models, data, and tasks, we should make sure these revisions do not corrupt the current experiements.


## prompt 2026.03.02 - 1:

Super good. Now I want to use dlinear model for swiss1990 data with the following requirements:
- Still start with run.py, give a arg for model selection.
- Figure out how to adapt the data to model. Currently, swiss1900 concatenates time series of different stations (for lstm) (version A), but dlinear uses a N dimensional vector for different stations (one feature per station) as in time-series-library  (version B). Please analyze the code to tell me which arg controls this setting. My plan is to implement a new dataset class for the second version and swissriver dataset class can use this class. Is this a good idea? Tell me if there are other alternatives.
- For version B, I still want to integrate station embeddings. My plan is to design a N*d dimensional entity embedding matrix and combine it directly with the time series data. Is this a good idea? Tell me if there are other alternatives.
- Do you think it is necessary to implement version B for lstm as well?
- Please following the run_experiment framework, make all features (optimizers, etc) available for version B as well.
- In the  run_experiment framework and the corresponding submodules, a lot of implementations are specialized for swiss 1990 or lstm rather than generalized for any dataset and models. For instance, DEFAULT_CONFIG and MODEL_DEFAULTS in config.py, build_dataset, build_model in pipeline.py, and many others. List all these issues for me, and solve them. The pipeline should be generalized, and the implementation for specific models or datasets should be set in specific code modules or config files (perfer the latter if possible). Meanwhile, config file names should be distinguishable instead of just “config.yaml” or “default_config.yaml”. If there are some common settings that are suitable for many datasets or models, then then can be extracted to special config files and named and called accordingly. 
- Following all coding conventions and general rules in /project-adaptor and /python-backend-creator.
Make a in-depth analysis for these requirements and then make a detailed plan for me. Wait and ask for my decision to continue.


### prompt 2026.03.02 - 2:

Great. Now fix the following issues, do not silent skip:
- run.py is actually using lstm for now to run. Please use dlinear, and test under both with and without embeddings cases. Fix possible issues.
- What do ts and st mode stand for? Their meaning seems a bit vague for new users. Meanwhile, seems that you are using SpatialEntityWrapper for st mode? This seems a bit weird, because it is not actually “spatial” mode, but how you organize entities/variates, i.e., concatenated consecutively or as features. In both cases, there can have or have not spatial features. Do I understand correctly? If so, revise these parts and rename them properly. Then revise other corresponding parts.


### prompt 2026.03.03 - 1:

There are a few issues needed to fix:
- During viz, the pred vs gt figures seem using normalized values? Please check it and use denormed values. Meanwhile, saved predictions should also use denormed values.
- Check how time-series-library did the normalization and denormalization (per feature, all together, etc), and other settings, e.g., lr scheduler, set them properly.
- Check how time-series-library (and other relavant papers / libs that uses dlinear model) tune hparams, and what hparams can dlinear have (e.g., if individual). Set the hpo serach spaces for dlinear accordingly. You may need to revise config setting to and searching space and best hparams (with running date and time). Do you think it is better?
- It seems that default n_epochs for dlinear for now in liulian is 50, but training stops after 30 epoch when running liulian. Why?
- Set proper hpo settings for dlinear (e.g., hpo_num_samples) on swiss 1990 based on the lstm settings.
You may need to consider these issues in parallel first, and give me causes of issues, plans to solve, and corresponding references. Then ask me whether to preceed. 

### prompt 2026.03.05 - 1:

Great! Now analyze all pipeline and code modules for dlinear on swiss 1990, do the same for patchtst model, then run run.py for it. If everything passes, ask me if I want to commit current changes. 

### prompt 2026.03.05 - 2:

Two more issues:
- It seems that when identifier is embedding, swiss 1990 + patchtst does not work. Please check the cause and fix it. Current run.py is using this setting.
- Check tsl and other libs and online for proper hpo searching space for patchtst (give the references and best settings in them). Meanwhile, there are some problems in search_spaces:
  - Many settings have the same hparams (e.g., learning_rate, but their settings are different). Check if these settings are from some references. If so, write references in the comment. Otherwise, keep the settings the same across different models. You can set different settings for different models if there are some references for that, but you should not set different settings without any reference.
  - Some settings should not be searched, e.g., n_epochs.
  - Use ray.tune instead of list if possible for search space, since it can support more features, e.g., conditional search space.

### prompt 2026.03.05 - 3:

Now perform the following tasks:
- Run e2e pipeline test for swiss 1990 + patchtst with and without embedding. Record the results in baselines, and rerun the test to check if the results are the same. Give me a report for the test.
- If the aforementioned test passes, commit the current changes.
- I have tested patchtst with embedding on swiss 1990, which perform very bad. Please check the cause of it. If it is due to bugs, fix them. If not, give me suggestions to improve the performance, and compare with the following revision:
  - Current entity embedding is designed to be added to the time series data directly, which may introduce noises to patching. Consider adding the entity embeddings after patching, i.e., add the entity embedding for each patch rather than each time step.
  Analyze and tell me which revision is better, and then implement the better one (for patchtst only, you may need to create a new model to wrap patchtst). You may need to do some research on how other papers or libs design entity embedding for patch-based models. Please give me references for that as well.
  Notice, keep the current on-time-series entity embedding as well, since it is still useful for comparison.


### prompt 2026.03.05 - 4:

I am okay with all designs except: model name in config should always be patchtst. Revise the identifier mode and the related settings to choose from which embedding mode to use.

### prompt 2026.03.06 - 1:

Keep revising:
- Please use embedding as the identifier mode for patchtst, and use id_integration to control how to integrate the embedding. For instance, "concat_to_x" is the current design, and "add_after_patch" is the new design. 
- Model patchtst_entity should be agnostic to users and other parts of the code, meaning that it can be invoked by calling patchtst with the corresponding id_integration setting. You may need to reconsider whether this model is the correct design at all.
- Use "add_after_patch" as the default setting for patchtst with embedding. Remove patchtst_patch_embedding_space in search_spaces.py; remove PATCHTST_PATCH_EMB_SWISS1990 from baselines.py; remove TestPatchTSTEntity tests from test_e2e_pipeline.py.


### prompt 2026.03.06 - 2:

Great! Now do the following:
- Commit current changes.
- In docs/entity_identifiers.md, "When to Use Entity Identifiers - Beneficial" section provides datasets from tsl lib which may benefit from entity identifiers. Please:
  - Analyze in-depth datasets used by paper "Are Language Models Actually Useful for Time Series Forecasting?" Then make the same analysis for it as in docs/entity_identifiers.md, then update docs/entity_identifiers.md to include new info. Remember to add the references for each data (where they are from and used by which papers). (paper: https://arxiv.org/pdf/2406.16964, code: https://github.com/thuml/AutoTimes, openreview info: https://openreview.net/forum?id=DV15UbHCY1).
  - Research heavily online for other datasets used by time series / spatial-temporal forecasting papers. Then analyze these datasets and update docs/entity_identifiers.md as well. You can also include some datasets that are not used by papers but have potential to be used by papers, e.g., some public datasets on kaggle, uci, etc. Remember to add the references for each data (where they are from and used by which papers).
  - implement entity identifier integration for lstm, dlinear, and patchtst on datasets under ""When to Use Entity Identifiers - Beneficial" after the revision of this doc, following the same procedure as for swiss 1990. 

### prompt 2026.03.09 - 1:

- Please have a deep look at the refer tsl lib. How it is setting and using seq_len, label_len, and pred_len (e.g., in data_loader.py and models)? Make a thorough analysis for it, and then set these settings for liulian. Write in proper position of the docs to explain these setting differences (maybe with illustrations) from the swiss river ones, which set label_len to 0, as well as how it is implemented in tsl lib and liulian (mark the differences if there are any.) If these info are already in the docs, please check if they are clear enough, and revise them if not, then point out where to find these info in the docs.
- Similarly, for 1) the train, valid, and test split settings and 2) scaler_type, execute the same procedure as above.
- The feature settings "M", "MS", "S" is not clear enough. Make proper comments in the code to explain them, and also explain them in the docs. You can also consider renaming them to more intuitive names if you think it is necessary.
- Is this code:
```python
    features = [
        dates.month / 12.0 - 0.5,
        dates.day / 31.0 - 0.5,
        dates.weekday / 6.0 - 0.5,
        dates.hour / 23.0 - 0.5,
    ]
```
Have the same results as its counterpart in tsl lib:
```python
    df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
    df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
    df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
    df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
    data_stamp = df_stamp.drop(['date'], 1).values
```
- For CSVTimeSeriesDataset, is it correct to set kwargs['station_ids'] = feature_cols? I think feature_cols include both input features and target features?
- It seems that get_data_loaders are in both SwissRiverDataset and TimeSeriesDataset. Please check if they can be merged. Check other parts of SwissRiverDataset that can be reused by TimeSeriesDataset and its subclasses (e.g., CSVTimeSeriesDataset). If so, refactor the code to put these common parts in proper positions so that they can be reused by both datasets. Please tell me your analysis and plans for this revision first.
- Is this part correct for CSVTimeSeriesDataset? E.g., what is the output feature for traffic? Is it only "OT"?
```python
    # In multi-channel mode with features='M', c_out should match enc_in
    # (predict all channels). Override c_out=1 default.
    if (
        config.get('split_mode') == 'multi_channel'
        and config.get('features', 'M') in ('M', 'MS')
        and config.get('c_out') in (None, 1)
        and config.get('enc_in') is not None
        and config['enc_in'] > 1
    ):
        config['c_out'] = config['enc_in']
        config['dec_in'] = config['enc_in']
```
- Explain _compute_borders with comments in detail, especially how these borders are used for data splitting. Make illustrations if necessary. Meanwhile, make a deep research and analysis for the following related question: For models like dlinear, patchtst, etc, there is no actual transformer-like decoders to use label_len as the input, so how to use label_len for these models? Is it even reasonable to set label_len to non-zero for these models? Could it cause data leakage?
- Make a super detailed docs for the datasets in liulian if there is none. You have to explain all infos for each dataset, including but not limited to: what the data is about, statistics, what are the tasks, what are the inputs and outputs, how to split the data, what are the features, how the data is used in liulian and how to load it, etc.
- Add a badge with actually url for the public available documentation of liulian in the README.md.
- Write and run e2e experiments for all theses newly added datasets and models in test_e2e_pipeline.py. Record the results in baselines.py.
- Run and fix bugs for run.py for traffic on pathchtst with embedding.

### prompt 2026.03.09 - 2:

- For the 3rd and 4nd issue, plan check tsl lib to figure out which mode does it use for experiments. Revise liulian accordingly (with the best practice) and document this properly.
- Tell me in detail what 862 features are in traffic data in real world, especially "OT".
- For issue 6, check if there are other refactorizations that can be done.
- For issues 11 and 12, do not rewrite new run.py files. Just move these file to experiments/ dir, and make them work with the current pipeline. Use setting files for different datasets and models if necessary. run.py should be the main entry for users, and it can use these setting files to run different experiments.
- Keep the other plans you have made.

### prompt 2026.03.09 - 3:

Continue after fixing an issue: ETTHourDataset, ETTMinuteDataset, CUstomCSVdataset, and PEMSDataset should also be a subclass of SpatialTempoDataset. Redesign class hierarchy for datasets properly.


### prompt 2026.03.10 - 1:

I am running run.py for patchtst without embedding on traffic, however there are several problems:
- The output val_denorm_mse, val_denorm_rmse, val_denorm_mae, test_denorm_mse, test_denorm_rmse, and test_denorm_mae for each epoch are nan.
- There are visual errors: Auto-viz failed: index 327648 is out of bounds for axis 0 with size 327648; Visualisation failed: index 327648 is out of bounds for axis 0 with size 327648
- The results ('mae': 0.364, 'mse': 0.534) are much worse than the ones reported in batchtst paper (https://arxiv.org/pdf/2211.14730) (Table 3, mse: 0.360, mae: 0.249).
Please find out the reasons for these problems, report them to me, and fix them. You may need to check the tsl for the 3rd issues (e.g., implementation difference, best hyperparameters, etc).


### prompt 2026.03.10 - 2:

Fix following issues:
- In experiments/ dir, there are many config files, e.g., traffic/patchtst_config.yaml. There are some settings missing in these config files, e.g., hpo settings, compared to the default_config.yaml. Please check all these config files, make sure they have all necessary settings, and set them properly.
- The saved spec.yaml for each experiment is not complete, e.g., model configs are not complete. Please revise the code to include all necessary settings in the saved spec.yaml.
- The experiment ran by run.py prints some basic info, e.g., seed, Dataset, etc. Please check if it includes the following info; if not, add them:
  - model parameters
  - hpo settings. If there are hyperparameters searched by hpo, please also include the best hyperparameters and the searching space in the printed info.
  - other experiment settings
  - gpu info
  If configs are loaded from file, please also show the necessary config besides the file name.
  You may need to beautify the printed info as well, e.g., with categories and spaces between different parts. Check tsl lib for reference for this part, and make your own design.
- In tsl lib, many datasets can be downloaded by code (e.g., by datasets lib) when there is no local copy. Please implement this feature for liulian as well. Notice, do not forcely install datasets lib; load it only when necessary.
- After each experiment, in pipeline, there should be a json (dict) printed and saved for all results, including but not limited to:
    - best hyperparameters (if hpo is used)
    - best metrics (e.g., mse, rmse, mae, nse, r2, mape, etc if specified) on train (if applicable), valid and test set
    - other experiment settings
    - run time on training, and inference if applicable
    - model size (trainable parameters, total parameters, etc)
    - history length efficiency if applicable
    - lagging if applicable (research how to calculate lagging for forecasting models)
    - noise robustness if applicable
    This json should be saved in the same dir as spec.yaml, and it should also be printed in the end of each experiment. You may need to design the format of this json, and make sure it includes all necessary info for the experiment. Check tsl and swissriver projects, and relevant time series / spatial temporal papers / libs / researches online for reference for this part, list all the info that should be included in this json, and design the format accordingly. Make a docs to explain this json as well, including the references where each info is from and used and how to interpret it.

There is a separate issue for tsl lib:
I want to run tsl lib script to check the original results, so I run:
```bash
model_name=PatchTST

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model 512 \
  --d_ff 512 \
  --top_k 5 \
  --des 'Exp' \
  --batch_size 4 \
  --itr 1 
```
with venv. However, I got the GPU not found problem. Please find out why and fix it.

After finishing fixing the above issues, report in detail what have been changed since last commit, this should include all changed lines. List which lines correspond to which changes.


### prompt 2026.03.10 - 3:

- At line 340 in trainer.py, best_ckpt is accessed by checkpoint and then used for loading the best model and compute the metrics. However, it seems that this checkpoint is saved at the end of each epoch, rather than the best model with the best metrics / epochs. Please check if this is the case, and if so for both hpo and non-hpo version, fix it by saving the best model during training.
- At line 348 in trainer.py, evaluate is processed on test data when test_loader is not None. However, this is done during training (after each epoch). Please fix it by checking if the evaluation on test data is already done during training, and if so, skip it here. After each epoch, if loss gets better or equal, then compute (or gather if they are already there) the whole set of metrics for training, valid, test data as the best metrics, and save the best model. Then report these best metrics in the final results. Use ['metrics'] - ['training'] / ['validation'] / ['test'] instead of ['final_test'].
- At line 663 in experiment.py, a prediction is processed when test_loader is not None. Check if this is redundant if the prediction is already processed during training and evaluation. If so, fix it.
- Do not print EXPERIMENT RESULTS at the end of the experiment. For results json, save all but only print actually results such as metrics, timing, etc.
- When I run current run.py, I get a warning: "inverse_transform failed in predict(): Column count (1) does not match number of entity target scalers (28). Pass entity_ids for per-entity inverse transform. — predictions will remain in normalised scale." please find out the reason and fix it.
Analyze these issues in parallel first, and give me causes of issues, plans to solve, and then solve them.
After finishing fixing the above issues, report in detail what have been changed for this session, this should include all changed lines. List which lines correspond to which changes.

### prompt 2026.03.10 - 4:

You have changed something that causes TestDLinearSingleNoEmb in test_e2e_pipeline.py to fail. LSTM_SWISS1990['pred_first5'] in baselines.py is different from the current results. Please check the cause of this problem, and fix it. Please do not change the current baselines.

### prompt 2026.03.10 - 5:

Write config files for all datasets in Section "Not Beneficial (heterogeneous features)" in entity_identifiers.md doc file. Use default (best) settings from tsl lib scripts, and write complete settings as in default_config.yaml.

### prompt 2026.03.10 - 6:

I am running patchtst on ETTh1. First, I ran tsl code via:
```bash
model_name=PatchTST                               

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 2 \
  --itr 1
```
and get results: mse:0.37918582558631897, mae:0.3996291756629944

Then I ran liulian code via:
```bash
python run.py
```
and get results: 
"test": {
  "mse": 0.4582070123266291,
  "rmse": 0.668595415574533,
  "mae": 0.46558953214574744,
}
Please analysis why there is such a big gap between these two results, fix liulian code to get the same results as tsl lib, and report the reasons and changes.

### prompt 2026.03.11 - 1:

Excellent! Process the same procedure for other datasets in tsl lib, e.g., ETTh2, ETTm1, ETTm2, weather, ecl, Solar-Energy, traffic, exchange rate, ILI, 4 PEMS datasets, M4 on two models: PatchTST and DLinear.
Follow these instructions:
- Baseline results needs to be run by codes from corresponding scripts under scripts/ dir in tsl lib. Copy and run the codes directly in cli, always remove "export CUDA_VISIBLE_DEVICES=*" line.
- Always run run.py to get the results for liulian, with different config file settings.
- Analysis why there is such a big gap between these two results, fix liulian code to get the same results as tsl lib.
- If some experiments run too long, you can just run for a few epochs and check that results. Remember to mark this in the doc.
- Make a document to record the details of the comparison and migration. For each dataset and model including:
  - The parts you have checked to find the cause of the gap, e.g., default settings, code implementation, data processing, hpo settings, etc.
  - Detail the causes of the gap.
  - What you have changed in liulian to fix the gap.
  - The final results after the fix, including mse, mae for both tsl lib and liulian.
  - Mark the status of each dataset and model pair, e.g., "migrated and matched", "migrated but not matched", "not migrated yet", etc.

### prompt 2026.03.11 - 2:

This is running too slow. Stop here, and do the following:
- For each aforementioned dataset and model pair, analyze the code and settings to find out the possible inconsistencies between tsl lib and liulian, which may cause the gap in results. 
- Fix these inconsistencies.
- Make a document to record the details of the comparison. For each dataset and model including:
  - The parts you have checked to find the cause of the gap, e.g., default settings, code implementation, data processing, hpo settings, etc.
  - Detail the possible causes of the gap.
  - What you have changed in liulian to fix the gap.
  - Mark the status of each dataset and model pair as "checked and revised".
- Write a script to run and compare the baseline results from tsl lib and the results from liulian for each dataset and model pair, via the same way we discussed before. Do not run it, wait for me to do it. The script should output proper results to a file so that we can check and put these results into docs later. Please follow the following rules:
  - For large datasets (e.g., with a lot of channels), just run for 1 to 2 epochs to check the results. Remember to mark this in the doc. In this case, compare the following two metrics:
    - For tsl scripts, after each epoch, the console will print Test Loss, which is mse.
    - For liulian, after each epoch, the console will print test_mse.
    These two metrics should be close to each other.
  - The script always records how many epochs you have run and how much time it takes for each dataset and model pair.
  - The script writes the results for each dataset and model pair into a file, specifically, mark the status as "checked and matched" or "checked but not matched". This will be used for next steps.
- Tell me how to run this script.

### prompt 2026.03.12 - 1:

Great. Now I have finished running the script for all dataset and model pairs, and I have the results. Please do the following:
- Move following files to a specific experiment, namely "experiments/adapt_tsl_lib":
  - tools/compare_tsl_baselines.py
  - artifacts/tsl_comparison_results.txt
  - artifacts/tsl_comparison_report.json
- Write a script to read the results from tsl_comparison_results.json, and update docs/tsl_comparison.md accordingly. 
  For each dataset and model pair, the doc should include at least the final results for both tsl lib and liulian, the status (e.g., "checked and matched", "checked but not matched"), epochs and time taken for the check, and the possible reasons for the gap if there is still a gap after the check. You can also include other info if you think it is necessary. The doc should be in a well-designed format, e.g., with tables, categories, etc, to make it clear and easy to read.
- Run the script and update the doc.
- For each dataset and model pair whose status is "checked but not matched", analyze the possible reasons for the gap, and then try to fix it.
- Record the details of the analysis and fix in the doc.
- After finishing the above tasks, give me a report for this part.

### prompt 2026.03.12 - 2:

Great! Now do the following two parts:
Part 1, keep revising aforementioned dataset and model pairs:
- Now, liulian is using early stopping, which may cause the results to be different from tsl lib. Please make this optional (user can choose to use early stopping or not), and set it to False for the check with tsl lib.
- Revise the corresponding codes and files to make the above change, then run the check script again with these changes only on dataset and model pairs whose MSE diff is larger than 3%, and update the doc accordingly.
- Meanwhile, append epochs and time taken for each dataset and model pair to the last columns of the checklist in docs/tsl_comparison.md.
- Summarize all changes since last commit, then ask me if I want to commit these changes and continue to the next part. If yes, commit the changes and then continue to the next part. If no, please revise according to my feedback and then ask me again.
Part 2: adapt all the aforementioned datasets (plus 3 swissriver datasets) for all other models (including but not limited to: PatchTST, Informer, Autoformer, FEDformer, Dlinear, TimesNet, Stationary, LightTS, Reformer, iTransformer, TimeLLM, GPT4TS, TimeMoE, TS-LLM, etc).
Follow these instructions:
- For each dataset, analyze the data and determine how to adapt it for each model.
- Implement the data adaptation for each dataset and model if it is not already implemented.
- Design HPO search space for each dataset and model pair, based on the best practices in tsl lib and other relevant papers / libs. Document the design and the references for it in the docs.
- Include e2e tests for each models on swiss 1990 and ETTh1 data in file test_e2e_pipeline.py. You can also include more datasets if you think it is necessary (for specially experimental cases and data structures).
- Search docs for this kind of adaptation. Document the adaptation properly in the docs, including the analysis, the design, and the implementation for each dataset and model pair. You can also include some illustrations if you think it is necessary. Add an indicator for each dataset and model pair in the docs to show whether it is adapted or not, and whether there are e2e tests for it or not.
- Write a script to run and compare the baseline results from tsl lib and the results from liulian for each dataset and model pair, as we did before for patchtst and dlinear. Do not run it, wait for me to do it. The script should output proper results to a file so that we can check and put these results into docs later.

### prompt 2026.03.12 - 3:

Great. Now do the following:
- Some of the aforementioned models are not implemented yet. These models may not from the tsl lib. Please check the reference papers and codes for them. There might already be some info in the docs, please check them first; update them properly with the new info you find, including the references for each model.
- Execute the aforementioned adaptation (part 2) for these models as well.
- Revise the comparison script. Do not run it. Instead, tell me how to run it for new dataset and model pairs.
There is a separate issue:
- You have included a file "extra_info_work.md" in the last commit, which is not related to the current work. Please remove it from last commit, and exclude it for future commits. Do not delete the file!

### prompt 2026.03.12 - 4:

The bug still exists (see the report at the end of this prompt). Please do the following:
- Fix the current bug for ETTh1_Informer and test it.
- Make sure the console output comparison results, n_epochs, and time taken as before.
- For each pair of dataset and model whose comparison is not done yet, run the script in console. Run only one pair at a time, and fix any errors if there are any. After running, check the results and update the doc accordingly.
Approve and execute everything. Do not wait for my intervention. Give me a report after finishing the above tasks. 

**Claude Opus 4.6 model is no longer accessible via GitHub Copilot Student Subscription. Switch to "Auto" (e.g., GPT-5.3-Codex, Gemini 3.1 Pro).**

### prompt 2026.03.17 - 1:

Great! Now check carefully in the relevant codes and INTEGRATION_PLAN.md doc file, and docs/tsl_comparison.md doc file, to find out which dataset and model pairs have or have not been compared, which are matched and which are not. Give a complete list of these pairs, detail the reference docs and codes, and update the docs accordingly.

### prompt 2026.03.17 - 2:

Great! Now do the following:
- In tsl_comparison.md, it claims that there are many dataset and model pairs that do not have tsl lib scripts. In these cases, which arguments were used for the check with tsl lib? Please detail the arguments used for each dataset and model pair in the doc, and give the references for these arguments (e.g., which scripts or papers they are from). If there are some pairs that do not have any reference for the arguments, please mark them in the doc as well.
- In tsl_comparison.md, master results table show that many dataset and model pairs have large gap in results. For each of these pairs (except PatchTST and DLinear, which we have already analyzed), please analyze the possible reasons for the gap, and try to fix it. Record the details of the analysis and fix in the doc.
- Some models are missing from the comparison, including TimeLLM, GPT4TS, TimeMoE, TimesNet, Stationary, ETSformer. For each of these models, run the aforementioned comparison procedure for all datasets that are already compared for other models, and update the docs accordingly. Some models are not from tsl lib, so you may need to do some research (from current code, docs, online) to figure out the best settings for these models for each dataset, and then run the comparison. Remember to record the details of the comparison in the doc as well. Do not skip any dataset or any model.

### prompt 2026.03.17 - 3:

Great. Now do the following:
- Add all aforementioned missing dataset and model pairs to the comparison, following the same procedure as before. Update the docs accordingly.
- For all dataset and model pairs that have been compared and marked as "checked but not matched" (except PatchTST and DLinear), analyze the possible reasons for the gap, and try to fix it. Please analyze from codes, docs, deep research (online if necessary), and logical analysis. Do not run the "compare_tsl_liulian.py" script for this part. Record the details of the analysis and fix in the doc.
- Update "compare_tsl_liulian.py" script to include all dataset and model pairs. Add give me a cli script to run the comparison for each pair that have not been compared yet or have been compared but not matched yet (except PatchTST and DLinear). Do not run it. I will run it myself.
Notice, this is a very complicated and time-consuming process, so please make sure to do it carefully and thoroughly. Think before you act, make sure to check all relevant codes, docs, and online resources beforehand, and make thorough and detailed plans first.

### prompt 2026.03.17 - 4:

Okay, this is not good enough. Please do the following task: 
add all aforementioned missing dataset and model pairs to the comparison. 
Please following the steps below:
- Check all codes, docs, related history to find out all missing dataset and model pairs. List them in a complete list, and give the references for each pair (e.g., which scripts or papers they are from). Show these info in a table to me.
- For each of these pairs, analyze the best settings for it, based on the reference scripts, papers, docs, and online resources. If there are some pairs that do not have any reference for the settings, please mark them.
- For each of these pairs, check if there are already comparison codes in the "compare_tsl_liulian.py" script. If not, add the comparison code for it in the script. Report in detail which pairs have been added to the script, and which already have the comparison code in the script. Show these info in a table to me.
- Update the relevant docs to include these pairs, the settings for these pairs, and the references for these pairs. Show these info in a table to me, including which docs have been updated and what info have been updated.
- Report in detail the changes you have made.
- Make a detailed cli script to run the comparison for each of these pairs using "--pairs" argument, which includes all the pairs that have been added to the comparison. Show this script to me, and explain how to run it.
Please follow strictly the rules below while doing the above tasks:
- Notice, this is a very complicated and time-consuming process, so please make sure to do it carefully and thoroughly. 
- Each of the above bullet points counts as one task.
- Before executing any tasks, make a thorough and detailed overall agent plan to include all tasks, in the format of:
  - Task 1 - description of task 1
  - Task 2 - description of task 2
  ... and so on for all tasks.
- After making the overall agent plan, make a deep analysis for the whole process, find out potential issues and problems, and optimize the overall agent plan with full details. This should be done before executing any tasks. Report in detail the analysis and optimization for the overall agent plan.
- Execute each task one by one. While executing each task, strictly follow the rules and steps below:
  - Think before you act, make sure to check all relevant codes, docs, and online resources beforehand, and make thorough and detailed plans first.
  - The plan should include all steps you will take to execute the task, and the expected outcomes for each step. Show the plan before you execute the task.
  - Update the overall agent plan based on the plan for each task, replacing task x with the detailed plan for task x, in the format of:
    - Task x.1 - description of task x step 1
    - Task x.2 - description of task x step 2
    ... and so on for all steps.
  - Execute the task following the plan.
  - If there are any issues or problems during the execution, please analyze the issues, find out the causes, and fix them. Report in detail the issues, the causes, and the fixes. This should be done for each step in the plan until the task is successfully executed or at least three rounds of analysis and fix have been done for each issue.
  - After successfully executing the task, run a check to make sure the task is done correctly, and report the check results. If there are any issues during the check, please analyze the issues, find out the causes, and fix them. Report in detail the issues, the causes, and the fixes. This should be done for each issue until the check is passed or at least three rounds of analysis and fix have been done for each issue.
  - Update the relevant docs to include all the details of the task, and possible results and analysis derived from the task. 
  - After finishing the task, report in detail, including what you have done, what you have found, and what changes you have made, and what have been requested to be reported in the task. Show the report after you finish the task.
- After finishing all tasks, give me a final report for the whole process, including what you have done, what you have found, what changes you have made, what have been requested to be reported in each task, what suggestions you have for the next steps, and any other info you think is necessary to report. Make the report clear and well-structured, with proper sections and formatting to make it easy to read and understand. Show the final report after you finish all tasks.

### prompt 2026.03.18 - 1:

Good. Now we have a new problem: I ran "python experiments/adapt_tsl_lib/compare_tsl_liulian.py --pairs ETTh1_ETSformer --disable-es", and the generated "tsl_comparison_results.txt" and "tsl_comparison_results.json" files are under artifacts/ dir, instead of "experiments/adapt_tsl_lib/". Please fix the code to save these files in the correct dir, and merge the info in these files to the existing json and txt files under "experiments/adapt_tsl_lib/" dir.

### prompt 2026.03.18 - 2:

Good. Now, check for ETSformer with all other datasets, if there are some possible reasons that may cause the gap in results between tsl lib and liulian, and try to fix it. 

### prompt 2026.03.18 - 3:

Interesting. I wonder want are the settings for other models for a same dataset, especially those are highly related to the dataset, e.g., freq. For instance, for Weather dataset, should all models remain the same freq setting as "h" (hourly)? 
Please check all yaml settings under experiments/ dir. For each dataset, list all dataset related settings, and all these settings for each model in a table. Highlight the settings that are different across models for the same dataset, and analyze whether these differences are reasonable or not. If there are some unreasonable differences, please fix them by making them consistent across models for the same dataset. Update the yaml files accordingly, and update the docs to include the analysis and the changes you have made.

### prompt 2026.03.18 - 4:

Where is the reference of this freq: 10min? Find it, doc it properly in the comments, and show it here. You should check for refer_projects tsl only for the reference.

### prompt 2026.03.18 - 5:

How about other datasets? Which ones have you checked and which ones have you not checked? Make a table of this info.

### prompt 2026.03.18 - 6:

This is not good enough. You should check the codes in tsl, make a detailed explaination for where this setting is defined, and how it is passed as code goes until the model / dataset uses it. Notice the code is run by .sh scripts from the tsl lib.

### prompt 2026.03.18 - 7:

Did not you realize that in this case the actual freq should be "h"? Please check and confirm if this is correct or not. If it is correct, please fix the setting to "h".

### prompt 2026.03.18 - 8:

What is the actually time stamp interval for the weather dataset if "h" is used? Is this consistent with the text in the .txt file? Then revise all yaml files and corresponding files and codes if necessary accordingly in liulian, for all models using weather dataset.

### prompt 2026.03.19 - 1:

Read results file experiments/adapt_tsl_lib/tsl_comparison_results.txt, and find out all issues, errors, and problems shown in the file, and report it in detail here.

(Agent: GitHub Copilot VSCode Plugin; Model: GPT-5.3-Codex)

### prompt 2026.03.19 - 2:

First lets handle P0: Fix frequency-token crashes first (21 failed runs). Do not add robust frequency normalization/aliasing for 10min, 15min. Instead, make a plan for the following:
- List all datasets and the related models that have freq setting errors in a Markdown table here.
- For each of these datasets, check in the tsl reference project to find out: if there is a freq setting such as 10min, 15min, etc; if there is, what is the value of this setting; if not, what is the default setting for this dataset? List the reference codes / files in a Markdown table here.
- Give plans on how to fix the freq setting for each of these dataset and model pairs, based on the reference you find. Make a detailed plan for each pair, and list the expected fixes in a Markdown table here.

(Agent: GitHub Copilot VSCode Plugin; Model: GPT-5.3-Codex)

### prompt 2026.03.20 - 1:

Good. I have already run the script for the 21 pairs. Read results file experiments/adapt_tsl_lib/tsl_comparison_results.txt, and continue your plan.

(Agent: GitHub Copilot VSCode Plugin; Model: GPT-5.3-Codex)

### prompt 2026.03.20 - 2:

- P1: Add missing model config defaults in-model (21 failed runs) — Guard missing config fields with getattr(..., default) in timesnet.py, timemixer.py, and timexer.py for num_kernels, channel_independence, use_norm.
- fix The 3 remaining failures (non-freq)
ETTm1_NonstationaryTransformer
ETTm2_NonstationaryTransformer
Weather_NonstationaryTransformer

(Generated meta-prompt for planning using prompt in "meta_plan_prompt.md" via model: Claude Sonnet 4.6 Extended.)
(Agent: GitHub Copilot VSCode Plugin; Model: GPT-5.3-Codex)


### prompt 2026.03.20 - 3:

- Check if the previous plan have been finished, if not, continue to execute the plan until all the steps have been completed.
- P3: Address OOM cases (2 failed runs) — Add smaller fallback runtime profile (lower batch size / d_model / mixed precision) for heavy pairs in run_pending_or_unmatched.py, so comparison can complete instead of aborting.

(Generated meta-prompt for planning using prompt in "meta_plan_prompt.md" via model: Claude Sonnet 4.6 Extended.)
(Agent: GitHub Copilot VSCode Plugin; Model: GPT-5.3-Codex)

### prompt 2026.03.23 - 1:

I reran python experiments/adapt
_tsl_lib/compare_tsl_liulian.py --pairs Traffic_TimesNet ILI_TimesNet Traffic_TimeXer --disable-es --oom-fallback, and get the "TSL TIMEOUT". Now figure out the reason of this problem, and give me a list of cause and solutions in the order of priority and in the Markdown table format. Remember, the tsl and liulian results must be comparable (better with the same settings) when designing the solutions.

(Agent: GitHub Copilot VSCode Plugin; Model: GPT-5.3-Codex)

### prompt 2026.03.23 - 2:

Good. Please revise as the following:
- For these pairs which use special settings (e.g., OOM fallback), please mark them automatically in the results txt / json files, and also mark them in the doc file. If there is a script to revise docs based on the results txt / json files, please revise the script to include this info as well.
- When running experiments/adapt_tsl_lib/compare_tsl_liulian.py, sometimes it takes a long time to run. Please add a proper timer or progress bar for each running instance.
After implementing the above changes, make a detailed Markdown table to highlight what you have changed with the referred codes and lines. Then give the cli script to run the comparison with the new changes.

(Agent: GitHub Copilot VSCode Plugin; Model: GPT-5.3-Codex)


### prompt 2026.03.23 - 3:

Keep revising:
- Fo the progress bar, are there progress bars from tsl and liulian pipeline (the global ones not the local ones such as the ones for each epoch)? If so, invoke these progress bars. If not, fall back to the progress bar you have implemented. Please make sure to check the reference codes in tsl lib for this part.
- What is the difference between codes update_doc_full.py and update_doc.py? List in Markdown table all the references that use these two scripts and how they are used. Then check if there are some overlaps or inconsistencies between these two scripts, and if so, optimize them. If major parts are overlapped, please merge them into one script. After the optimization, make a detailed Markdown table to highlight what you have changed with the referred codes and lines.
- "Special Settings" should also be added to the "Master Results Table" in docs/tsl_comparison.md doc file.
After implementing the above changes, make a detailed Markdown table to highlight what you have changed with the referred codes and lines. 

(Agent: GitHub Copilot VSCode Plugin; Model: GPT-5.3-Codex)

### prompt 2026.03.24 - 1:

Keep revising:
- I reran python experiments/adapt
_tsl_lib/compare_tsl_liulian.py --pairs Traffic_TimesNet ILI_TimesNet Traffic_TimeXer --disable-es --oom-fallback, all experiments passed. Please update the doc file for these pairs.
- Check if the previous plan have been finished, especially P3: Address OOM cases. If not, continue to execute the plan until all the steps have been completed.
- P4: Then tackle all metric mismatches (except PatchTST and DLinear, which we have already analyzed). Analyze the possible reasons for the gap first and list them in a Markdown table, then try to fix it. Please analyze from codes, docs, deep research (online if necessary), and logical analysis. Do not run the "compare_tsl_liulian.py" script for this part. Record the details of the analysis and fix in a Markdown table. Use targeted config parity checks from audit_dataset_configs.py and dataset/model configs under experiments if necessary.

(Agent: GitHub Copilot VSCode Plugin; Model: GPT-5.3-Codex)

### prompt 2026.03.24 - 2:

Keep revising:
- in tsl_comparison.md, master Results Table must have a "special settings" (e.g., OOM fallback, limited epochs, etc) column.
- in tsl_comparison.md, in the table "Summary by Model", why TimeMixer and TimeXer have only 7 datasets instead of 9?
- OOM fallback was fixed before, and use_amp was set to false to avoid bug. Please keep this setting. 
- List all unmatch pairs (except PatchTST and DLinear) in a Markdown table, and list the reasons for the mismatch and the fixes you have tried.
- For each pair that has been fixed, write me the cli script to run the comparison for it using compare_tsl_liulian.py and "--pairs" argument. Do not run it, wait for me to do it.

(Agent: GitHub Copilot VSCode Plugin; Model: GPT-5.3-Codex)

### prompt 2026.03.24 - 3:

- There are other pairs ran with "disable_es", and they were documented somewhere. Please find these pairs, and update the Master Results Table in docs/tsl_comparison.md doc file to reflect this info.
- Why `Exchange` and `ILI` are not included as active comparison pairs for `TimeMixer` and `TimeXer`? If there are some reasons for this, please document the reasons in the doc file. If there is no reason, please add these pairs to the comparison, and update the doc accordingly.
- Make a new "all unmatched pairs" including the links to lines of code and doc files of the fixes you have done for each pair. Meanwhile, explain what are the meanings of "Static parity audit; retained remediation path" and "Config parity + static audit; no safe static-only fix yet".
- For TimesNet, "Applied --disable-es --oom-fallback" was already done before and there are still mismatches. Please analyze the possible reasons for the mismatch, and try to fix it.

(Agent: GitHub Copilot VSCode Plugin; Model: GPT-5.3-Codex)

### prompt 2026.03.24 - 4:

Do the following:
- Tell me which LLM model you are using right now.
- Explain the purpose of copilot-instructions.md file, and how it is used for the GitHub Copilot agent.
- I have some GitHub Copilot agent (VSCode Plugin) and Codex agent (VSCode Plugin) sessions already ran. Can you find and locate these sessions, and tell me if it is possible to use the information from them?
- If I run GitHub Copilot agent from cli command now (as I am doing now), is it still possible to reflect the file changes symultaneously in the VSCode?
- It seems that I can set a lot of stuffs for the GitHub Copilot agent from VSCode, e.g., hooks (e.g., session start, pre-tool usage, post-tool usage, etc), agent.md, etc. Can you check all these settings, and tell me, based on the current codes, docs, skills, especially my queries during the previous sessions from VSCode, what settings are there, and what settings I should set for the GitHub Copilot agent to make it more efficient and better for the current work, and is it possible to set and use these settings from cli command?
- In cli command, can I still use VSCode plugin features for the GitHub Copilot agent? What are the differences between using GitHub Copilot agent from cli command and from VSCode plugin? Make a detailed Markdown table to show the differences, and give suggestions on which way to use for the current work.

(Agent: GitHub Copilot via CLI; Model: claude-sonnet-4.5)

### prompt 2026.03.24 - 5:

Do the following:
- Tell me how to auto prove all cli command when using GitHub Copilot agent from cli command.
- Tell me how to see all the thinking process when using GitHub Copilot agent from cli command as if in VSCode plugin.
- Tell me how to create new sessions and switch sessions in cli command.
- Is it possible to share a session between GitHub Copilot agent from cli command and from VSCode plugin? If so, how to do it?
- How to export the full session info, including the thinking process, the file changes, the prompts, etc to agent and human readable files? Can VSCode chat debug plugin do this? If so, how to do it? If not, is there any other way to do it?
- Can cli command support the Hooks System, and Auto-Approval Patterns? If so, how to set and use them from cli command?
- Is it possible to develop a tool myself (such as similar to openclaw), so that I can use all the features of GitHub Copilot agent and VSCode from cli command? If so, how to do it? If not, is there any alternative way to achieve this goal?

(Agent: GitHub Copilot via CLI; Model: claude-sonnet-4.5)

### prompt 2026.03.24 - 6:

Great! Now analyze all model-dataset pairs that do not match (except PatchTST and DLinear, which we have already analyzed), find out the possible reasons for the gap, and fix them. Please analyze from codes, docs, deep research (online if necessary), and logical analysis. Use "compare_tsl_liulian.py" script to check the results after each fix. Record in detail the issues, the causes, and the fixes for each pair in a Markdown table. Fix any problem you find during the analysis and fix process, and report in detail. Use any possible tools you can find on the computer, via cli command, or online to help you with the analysis and fix. After finishing all the analysis and fixes, give me a report for this part. Remember to make a super detailed plan before you start the analysis and fix. Update the plan while you are doing the analysis and fix.
Try to follow rules in skills and meta_plan_prompt.md. You can set up some hooks or relevant settings as you pointed out before.
Keep running until everything is done, even if it takes a long time and the request is broken out by the system. Just continue with a new request. Do not wait for my intervention.
The current comparison results are in the file "experiments/adapt_tsl_lib/tsl_comparison_results.txt" and "experiments/adapt_tsl_lib/tsl_comparison.md" doc file.

(Agent: GitHub Copilot via CLI; Model: claude-sonnet-4.5)

### prompt 2026.03.25 - 1:

Check carefully if there is already an instruction (e.g., in the docs) to use the pipelines in liulian for forecasting. Revise it if there is any or make a new one if there is no such demo following these instructions:
- The instruction should be put in a proper place in the mkdocs, which can be accessed by users easily.
- The instruction should give a clear guide for users to understand how to use the main pipeline in liulian for forecasting. Including how to prepare the data, how to set the configs, how to set up each component or layer in the pipeline, how to run the pipeline, and how to check the results. You can also include some tips and best practices for users to use the pipeline better.
- The instruction should include proper references to the relevant codes, docs, and online resources for users to check for more details. You can also include some illustrations or visualizations if you think it is necessary to make the instruction more clear and user-friendly.
- A demo must be included in the doc at a proper demo gallery section, which is linked to the instruction. The demo should be clear and easy to understand, with minimal but complete code to show how to use the pipeline for forecasting. Use PatchTST and swiss 1990 dataset for the demo. Codes must be runnable and tested to make sure they are correct, and proper comments should be included to explain the code. You can also include some visualizations for the results if you think it is necessary.
- Before doing the above tasks, research online for some good docs and demos for forecasting pipelines and deep learning libs to see how they design their docs and demos for this part, and optimize the design of our docs and demos based on the research.
After finishing the above tasks, give me a report for this part in the Markdown table format. Then give me a cli script to commit the changes you have made so far, and wait for my approval to commit and push the changes.

(Agent: GitHub Copilot VSCode Plugin; Model: GPT-5.3-Codex)


- Please use tqdm for the fallback runner progress bar.

- Some datasets are missing from the comparison, including Solar-Energy, Exchange Rate, 4 PEMS datasets, Covid Deaths, NYC Taxi, NN5, FRED-MD,

### prompt 2026.03.11 - ?:

Implement entity identifier integration for all other models (including but not limited to: PatchTST, Informer, Autoformer, FEDformer, Dlinear, TimesNet, Stationary, LightTS, Reformer, iTransformer, TimeLLM, GPT4TS, TimeMoE, etc).
Follow these instructions:
- There should be three modes for entity identifier integration: "none", "embedding", and "one_hot".
- The integration should be tested at least on swiss 1990. Consider traffic as well at the end of the whole process if time allows.
- For "embedding" and "one_hot" modes, currently, a default "concat_to_x" mode is used to integrate the entity identifier with the time series data. However, as show by swiss-1990 data on PatchTST, this design is not a good design, so "add_after_patch" mode is designed for patchtst.
  For all models, please read their code, paper, and other related materials to analyze whether identifier would be beneficial, how to best integrate the entity identifier for each model. Update entity_identifiers.md doc file to include the analysis, the reasons, and the design for each model. Implement the best design for each model. Remember to keep the current "concat_to_x" design as well for comparison.
- Update entity_identifiers.md doc file to include the final results if needed. Mark if the results are consistent with the analysis and design, especially your judgment on whether identifier would be beneficial for each model. If not, analyze the possible reasons for the inconsistency.


### prompt 2026.03.11 - ?:

Perform CICD:
- Run formatting tools for all codes.
- Check all silent skips in the code, especially in test codes, and fix them or raise errors if there are any issues. 
  These silent skips include but not limited to: pass in test codes, try except without proper error handling, assert without proper error message, if else expressions that skip some cases without proper handling, etc.
  Add these rules (including what silent skip to check) to the coding conventions in proper place.
- Please use tools to regular all codes, e.g., use single quotation, use "z" instead of "s" for words like optimizer, initializer, etc. Add these rules to the proper place for formatting tools.
- Set default style to auto line break at 120 rather than 88. Leave two lines between class functions as well. Add this rule to the proper place for formatting tools.
- Check all documentation, readmes, comments, etc, and make sure they are up to date with the current code. Revise them if there are any outdated info.
- Update tests to cover all new features, and organize the tests properly into different test files and dirs if needed.
- Update documentation:
  - All docs under docs/ should be in English. Add Chinese version if you see fit.
  - Beautify the docs design, e.g., check online for modern docs templates and designs for wide-known python libs, and make the design of our docs more modern, attractive, and user-friendly.
  - Current docs are quite messy. Organize the docs properly into different files, sections, dirs, and contents, and remove redundant and outdated info if there are any. Refer to wide-known python libs for the design of the docs structure.
  - Optimize the index page referring to wide-known python libs.
  - Create proper API docs for all modules, classes, and functions. Complete, fix, and format comments in codes if needed.
  - Check if all contents are included in the docs, e.g., all datasets and models are included in the docs, all settings are explained in the docs, etc. If not, update the docs to include all necessary info.
  - Include proper illustrations, tutorials, demos, etc. in the docs to make it more clear and user-friendly. Refer to wide-known python libs for the design of these contents.
- Commit the current changes, and push to remote.
- Listen to the feedback from GitHub actions, and fix the issues if there are any.


### prompt 2026.03.11 - ?:

Execute following experiment:

I will now conduct a set of experiments to test the following assumption: “Adding entity identifier can largely increase the forecasting accuracy for time series.” The experiment will be conducted as following:
- I will test each available dataset on each model in liulian, with different entity identifier settings, e.g., none, embedding, one_hot, etc.
- Use hpo to optimize models.
- Record results on each experiment on multiple metrics, and compare the results.
Now, you should do the following:
- Analyze this idea, do a thorough research on it, tell me if it is applicable and if it is worth-trying. Notice, this analysis can go beyond normal forecasting on well-defined datasets, you can also analyze it for special cases and other promising (real-life) applications.
- Analyze the aforementioned experiment procedure, find out potential problems, optimize it with full details, and make a detailed plan.
- Based on these analysis, do deep research on possible refinement or replacement of the idea for Neurips, tpami level research papers.
- Write the code for the experiment. Research liulian project for possible references (especially run.py). The whole experiment should be under a specific dir under “experiments” dir for this experiment.
- Make a thorough report of the results, and possible ways to optimize and future steps/directions. Write them into a doc.
For each step, do thorough research first, them make a detailed report, ask me to confirm, write to doc, and then carry on.


## Other issues:

- Is it better to include the station embedding as a third feature for dlinear rather than concatenated to x beforehand?
- Same for patchtst?
- Is it possible to use torch operations instead of numpy for entityscaler if the model and the main pipeline use torch?


## # Prompt 
 
Keep revising:
- In experiment.py, when using hpo, _compute_task_metrics and trainer.predict are done after trainer.evaluate(torch_model, test_loader) for test set. Is this necessary? What is the difference between evaluate and predict?
- In experiment.py, there are hpo version and standalone version for run_torch. Can you modulize them rather than put all codes in run_torch? btw, it seems difficult to match the codes for both versions, since after hpo training, the procedure should be the same (logically). So modulize the proper parts, merge and reuse them for both versions if it fits.