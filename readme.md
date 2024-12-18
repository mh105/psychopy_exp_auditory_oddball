# Auditory oddball task
Last edit: 11/22/2024

## Edit history
- 11/22/2024 by Alex He - removed summary csv saving since no trialList used
- 11/13/2024 by Alex He - added the ability to repeat the practice tones
- 10/24/2024 by Alex He - added a print message of task ID at the onset of task
- 10/12/2024 by Alex He - increased logging granularity from warning to debug (maximal level)
- 10/10/2024 by Alex He - added MilliKey response box and finalized voice-over audio
- 09/25/2024 by Alex He - added winHandle.activate() to make sure window is on foreground
- 09/23/2024 by Alex He - upgraded to run on PsychoPy 2024.2.2
- 09/05/2024 by Alex He - removed git tracking of _lastrun.py file and added retries to pyxid2.get_xid_devices() with timeout
- 08/17/2024 by Alex He - added more print messages during c-pod connection
- 08/12/2024 by Alex He - reverted to python 3.8 as pylink connection to EyeLink does not work correctly on 3.10
- 08/04/2024 by Alex He - generated experiment scripts on python 3.10
- 08/02/2024 by Alex He - upgraded to support PsychoPy 2024.2.1
- 07/23/2024 by Alex He - upgraded to support PsychoPy 2024.2.0
- 07/17/2024 by Amber Hu - added the ITI variable into output file
- 06/30/2024 by Alex He - created finalized first draft version

## Description
This task is used to elicit the classical P300 event-related potential (ERP) response to oddballs in a target detection paradigm. A number of variations of the task design have been used in the literature as well summarized by:

Polich, J. (2007). Updating P300: an integrative theory of P3a and P3b. Clinical neurophysiology, 118(10), 2128-2148.

We have chosen the auditory modality as subjects listening to tones with eyes closed to reduce the chances of eye blink and movement artifacts in the EEG recordings. We have applied state-space ERP extraction to pilot data of this behavioral paradigm, and we obtained robust potential responses with as few as 60 trials. Therefore, instead of repeating for two blocks of 100 trials each, we are only administering a single block of 100 trials.

Past literature has disagreed on whether behavioral responses should be elicited for both targets and distractors, as well as on the percentage of oddballs. We have decided to use 20% oddballs to increase the rarity and strength of the P300 response; for the same reason, subjects are only responding to the target tones (oddball) and not producing any behavioral responses for distractor tones (regular) - this may cause an additional difference between target and distractor trials as the presence of motor responses, but the P300 time course should still be largely driven by cognitive processes, and previous studies on P300 have used this response configuration. Since our primary goal of using this task is to elicit a canonical ERP response in normal aging, mild cognitive impairment (MCI) and Alzheimer's disease (AD) patients, we can tolerate the slightly reduced interpretability of the P300 ERP to a specific cognitive process.

## Outcome measures
- P300 ERP waveform derived from contrasting between target and distractor trials
