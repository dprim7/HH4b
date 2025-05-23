# Run-3

- To postprocess: `PostProcess.py`
- To create datacards: `CreateDatacard.py`
- To plot: `PlotFits.py`

### ANv1:
```/uscms/home/jduarte1/nobackup/HH4b/src/HH4b/postprocessing/templates/Apr18
```
made with:
```
cd postprocessing
python3 PostProcess.py --templates-tag Apr18 --tag 24Mar31_v12_signal --mass H2Msd --no-fom-scan --templates --bdt-model v1_msd30_nomulticlass --bdt-config v1_msd30
python3 postprocessing/CreateDatacard.py --templates-dir postprocessing/templates/Apr18 --year 2022-2023  --model-name run3-bdt-apr18
```
Fits:
```
cd cards/run3-bdt-apr18
run_blinded_hh4b.sh --workspace --bfit --limits --dfit --passbin=0
python3 postprocessing/PlotFits.py --fit-file cards/run3-bdt-apr18/FitShapes.root --plots-dir ../../plots/PostFit/run3-bdt-apr18 --signal-scale 10
```

### ANv1
```
python3 PostProcess.py --templates-tag May2 --tag 24Apr23LegacyLowerThresholds_v12_private_signal --mass H2Msd --no-legacy --bdt-config v1_msd30_txbb  --bdt-model v1_msd30_nomulticlass  --no-fom-scan --templates --txbb-wps 0.92 0.8 --bdt-wps 0.94 0.68 0.03 --years 2022 2022EE 2023 2023BPix
```

### ANv2
Apr 22
```bash
python PostProcess.py --templates-tag 24Apr21_legacy_bdt_ggf --data-dir /ceph/cms/store/user/rkansal/bbbb/skimmer/ --tag 24Apr19LegacyFixes_v12_private_signal --mass H2PNetMass --bdt-model 24Apr21_legacy_vbf_vars --bdt-config 24Apr21_legacy_vbf_vars --legacy --no-fom-scan-bin1 --no-fom-scan --no-fom-scan-vbf --no-vbf
```
Frozen for ANv2
```bash
python3 PostProcess.py --templates-tag 24May9v2Msd40 --tag 24Apr23LegacyLowerThresholds_v12_private_signal --mass H2PNetMass --legacy --bdt-config 24Apr21_legacy_vbf_vars  --bdt-model 24Apr21_legacy_vbf_vars --txbb-wps 0.99 0.94 --bdt-wps 0.94 0.68 0.03 --no-bdt-roc --templates --no-fom-scan --no-fom-scan-vbf --years 2022 2022EE 2023 2023BPix --training-years 2022EE --no-vbf
```

### ANv9
```
python3 PostProcess.py --templates-tag 24June3NewBDTNewSamplesPtSecond250 --tag 24May24_v12_private_signal --mass H2PNetMass --legacy --bdt-config 24May31_lr_0p02_md_8_AK4Away --bdt-model 24May31_lr_0p02_md_8_AK4Away --txbb-wps 0.975 0.92 --bdt-wps 0.98 0.88 0.03 --vbf-txbb-wp 0.95 --vbf-bdt-wp 0.98 --no-bdt-roc --no-fom-scan --no-fom-scan-bin2 --no-fom-scan-bin1 --data-dir /ceph/cms/store/user/cmantill/bbbb/skimmer/ --method abcd --no-vbf-priority --vbf --no-fom-scan-vbf --templates --pt-second 250
```

```
python3 PostProcess.py --templates-tag 24June10NewBDTNewSamplesPtSecond300 -tag 24May24_v12_private_signal --mass H2PNetMass --legacy --bdt-config 24May31_lr_0p02_md_8_AK4Away --bdt-model 24May31_lr_0p02_md_8_AK4Away --txbb-wps 0.975 0.92 --bdt-wps 0.98 0.88 0.03 --vbf-txbb-wp 0.95 --vbf-bdt-wp 0.98 --no-bdt-roc --no-fom-scan --no-fom-scan-bin2 --no-fom-scan-bin1 --data-dir /ceph/cms/store/user/cmantill/bbbb/skimmer/ --method abcd --no-vbf-priority --vbf --no-fom-scan-vbf --templates --pt-second 300
```

To scan:
```bash
python3 PostProcess.py --templates-tag Apr22 --tag 24Apr23LegacyLowerThresholds_v12_private_signal --mass H2PNetMass --legacy --bdt-config 24Apr21_legacy_vbf_vars --bdt-model 24Apr21_legacy_vbf_vars  --fom-scan --txbb-wps 0.99 0.94 --bdt-wps 0.94 0.68 0.03 --no-control-plots --no-bdt-roc --no-templates --no-fom-scan-vbf --years 2022EE --method sideband
```

For tt corrections:
```bash
python PostProcessTT.py --templates-tag 24May24 --tag 24May25_v12_private_had-tt --data-dir ../../../../data/skimmer/ --mass H2PNetMass --legacy --control-plots --bdt-model 24Apr21_legacy_vbf_vars --bdt-config 24Apr21_legacy_vbf_vars --year 2022 2022EE 2023 2023BPix
```

# Run-2
```bash
python3 PostProcessRun2.py --template-dir 20210712_regression --tag 20210712_regression --years 2016,2017,2018
python3 CreateDatacardRun2.py --templates-dir templates/20210712_regression --year all --model-name run2-bdt-20210712 --bin-name pass_bin1
./run_hh4b.sh --workspace --bfit --dfit --limits
python3 PlotFitsRun2.py --fit-file cards/run2-bdt-20210712/FitShapes.root --plots-dir plots/run2-bdt-20210712/ --bin-name passbin1
```
