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