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






## Other issues:

- Is it better to include the station embedding as a third feature for dlinear rather than concatenated to x beforehand?
- Same for patchtst?

- set default style to auto line break at 120 rather than 88. Leave two lines between class functions as well.
- Is it possible to use torch operations instead of numpy for entityscaler if the model and the main pipeline use torch?



## # Prompt 
 
Keep revising:
- In experiment.py, when using hpo, _compute_task_metrics and trainer.predict are done after trainer.evaluate(torch_model, test_loader) for test set. Is this necessary? What is the difference between evaluate and predict?
- In experiment.py, there are hpo version and standalone version for run_torch. Can you modulize them rather than put all codes in run_torch? btw, it seems difficult to match the codes for both versions, since after hpo training, the procedure should be the same (logically). So modulize the proper parts, merge and reuse them for both versions if it fits.
- Please use tools to regular all codes, e.g., use single quotation, use "z" instead of "s" for words like optimizer, initializer, etc.

Experiment 1:

Prompt:

I will now conduct a set experiments to test the following assumption: “Adding entity identifier can largely increase the forecasting accuracy for time series.” The experiment will be conducted as following:
- I will test each available dataset on each model in liulian, with different entity identifier settings, e.g., none, embedding, one_hot, etc.
- Use hpo to optimize models.
- Record results on each experiment on multiple metrics, and compare the results.
Now, you should do the following:
- Analyze this idea, do a thorough research on it, tell me if it is applicable and if it is worth-trying.
- Analyze the aforementioned experiment procedure, find out potential problems, optimize it with full details, and make a detailed plan.
- Based on these analysis, do deep research on possible refinement or replacement of the idea for Neurips, tpami level research papers.
- Write the code for the experiment. Research liulian project for possible references (especially run.py). The whole experiment should be under a specific dir under “experiments” dir.
- Make a thorough report of the results, and possible ways to optimize and future steps/directions.
For each step, do thorough research first, them make a detailed report, ask me to confirm, and then carry on.