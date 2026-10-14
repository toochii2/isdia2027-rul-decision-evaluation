# Reproduction pipeline. Run from the repository root.
#   make unpack     decompress the committed model predictions into results/
#   make analysis   recompute every summary result from the predictions (minutes to ~1 h)
#   make figures    regenerate the paper and supplementary figures
#   make models     retrain every model from the raw C-MAPSS files (many CPU hours)
#   make all        unpack + analysis + figures

PY      ?= python
DATA    ?= data/CMAPSSData
SUBSETS := FD001 FD002 FD003 FD004
R       := results

.PHONY: all unpack models analysis figures numbers check-data clean

all: unpack analysis figures

check-data:
	@test -f $(DATA)/train_FD001.txt || (echo "C-MAPSS files not found in $(DATA); see data/README.md" && exit 1)

unpack:
	@for f in $(R)/predictions/*.json.xz; do xz -dkc $$f > $(R)/$$(basename $$f .xz); done
	@echo "Predictions unpacked into $(R)/"

# ---------------------------------------------------------------- model training (slow)
models: check-data
	@for s in $(SUBSETS); do \
	  $(PY) scripts/cv_tabular.py $(DATA) $$s $(R)/cv_$$s.json; \
	  $(PY) scripts/cv_tabular.py $(DATA) $$s $(R)/cv_P1_$$s.json 125 101; \
	  $(PY) scripts/cv_tabular.py $(DATA) $$s $(R)/cv_P2_$$s.json 125 102; \
	  $(PY) scripts/cv_cnn.py $(DATA) $$s $(R)/cnn_$$s.json 150; \
	  $(PY) scripts/nested_tabular.py $(DATA) $$s $(R)/nested_$$s.json; \
	  $(PY) scripts/nested_cnn.py $(DATA) $$s $(R)/nestedcnn_$$s.json 150; \
	done
	@for s in FD001 FD003; do $(PY) scripts/cv_tabular.py $(DATA) $$s $(R)/cv_cap150_$$s.json 150; done
	$(PY) scripts/cnn_floor_check.py $(DATA) FD001 $(R)/cnn_floor_FD001.json 150
	DROP=0 $(PY) scripts/cnn_floor_check.py $(DATA) FD001 $(R)/cnn_floor_drop0_FD001.json 150
	$(PY) scripts/cnn_floor_check.py $(DATA) FD001 $(R)/cnn_floor_ep300_FD001.json 300

# ---------------------------------------------------------------- analysis (from predictions)
analysis:
	$(PY) scripts/lenses_test_set.py $(R) $(SUBSETS)
	$(PY) scripts/trajectory_bands.py
	@for s in $(SUBSETS); do $(PY) scripts/decision_nested.py nested $(R)/decision_nested_$$s.json $$s; done
	$(PY) scripts/merge_json.py $(R)/decision_nested.json $(foreach s,$(SUBSETS),$(R)/decision_nested_$(s).json)
	$(PY) scripts/decision_nested.py nonnested:cv_ $(R)/decision_P0.json $(SUBSETS)
	$(PY) scripts/decision_nested.py nonnested:cv_P1_ $(R)/decision_P1.json $(SUBSETS)
	$(PY) scripts/decision_nested.py nonnested:cv_P2_ $(R)/decision_P2.json $(SUBSETS)
	$(PY) scripts/decision_nested.py nonnested:cv_cap150_ $(R)/decision_cap150.json FD001 FD003
	$(PY) scripts/decision_fixed_margin.py $(R) $(SUBSETS)
	$(PY) scripts/accuracy_table.py
	$(PY) scripts/diagnostics.py
	$(PY) scripts/supplementary_analyses.py
	$(PY) scripts/cnn_floor_summary.py

figures:
	@mkdir -p $(R)/figures
	$(PY) scripts/fig_trajectory_error.py
	$(PY) scripts/fig_error_vs_decision.py
	$(PY) scripts/supplementary_figures.py

numbers:
	$(PY) scripts/report_numbers.py

clean:
	rm -f $(foreach f,$(wildcard $(R)/predictions/*.json.xz),$(R)/$(notdir $(basename $(f))))
