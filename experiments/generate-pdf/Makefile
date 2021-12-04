CHANNELS = 32
PROTOCOL = simple

all: results-adam-2-gpus.pdf results-lamb-2-gpus.pdf results-adamfp16-2-gpus.pdf results-lambfp16-2-gpus.pdf \
results-adam-4-gpus.pdf results-lamb-4-gpus.pdf results-adamfp16-4-gpus.pdf results-lambfp16-4-gpus.pdf \
results-adam-8-gpus.pdf results-lamb-8-gpus.pdf results-adamfp16-8-gpus.pdf results-lambfp16-8-gpus.pdf \
results-adam-32-gpus.pdf results-adam-16-gpus.pdf results-lamb-32-gpus.pdf results-lamb-16-gpus.pdf \
results-adamfp16-32-gpus.pdf results-adamfp16-16-gpus.pdf results-lambfp16-32-gpus.pdf results-lambfp16-16-gpus.pdf \
all-64-gpus \
all-interval-ranges \
scattered-results.pdf

all-64-gpus: results-adam-64-gpus.pdf results-lamb-64-gpus.pdf results-adamfp16-64-gpus.pdf results-lambfp16-64-gpus.pdf
all-256-gpus: results-adam-256-gpus.pdf results-lamb-256-gpus.pdf results-adamfp16-256-gpus.pdf results-lambfp16-256-gpus.pdf

all-interval-ranges: adam-interval-ranges.pdf adamfp16-interval-ranges.pdf lambfp16-interval-ranges.pdf lamb-interval-ranges.pdf

#FP32
results-adam-%-gpus.csv: dsl_vs_apex.py process_hypercluster_data.py
	python3 dsl_vs_apex.py --optimizer adam --ranks $* --channels $(CHANNELS) --protocol $(PROTOCOL)  > $@

results-lamb-%-gpus.csv: dsl_vs_apex.py process_hypercluster_data.py
	python3 dsl_vs_apex.py --optimizer lamb --ranks $* --channels $(CHANNELS) --protocol $(PROTOCOL) > $@

#Mixed Precision
results-adamfp16-%-gpus.csv: dsl_vs_apex.py process_hypercluster_data.py
	python3 dsl_vs_apex.py --optimizer adam --ranks $* --fp16 --channels $(CHANNELS) --protocol $(PROTOCOL) > $@

results-lambfp16-%-gpus.csv: dsl_vs_apex.py process_hypercluster_data.py
	python dsl_vs_apex.py --optimizer lamb --ranks $* --fp16 --channels $(CHANNELS) --protocol $(PROTOCOL) > $@

# results-adam-16-gpus.pdf: results-adam-16-gpus.csv plot_dsl_vs_apex.py
# python plot_dsl_vs_apex.py results-adam-16-gpus.csv

all-64-subplots.pdf: results-adam-64-gpus.csv results-lamb-64-gpus.csv results-adamfp16-64-gpus.csv results-lambfp16-64-gpus.csv
	python3 all_subplots_dsl_vs_apex_64_GPUs.py $^

%.pdf: %.csv plot_best_dsl_vs_apex.py plot_dsl_vs_apex.py
	python3 plot_dsl_vs_apex.py $<

%fp16-interval-ranges.pdf: interval_ranges.py
	python3 interval_ranges.py --optimizer $* --fp16 --channels minimum --pdf $@

%-interval-ranges.pdf: interval_ranges.py
	python3 interval_ranges.py --optimizer $* --channels minimum --pdf $@

scattered-results.csv: dsl_vs_apex.py
	python dsl_vs_apex.py --optimizer all --ranks 64 --channels minimum --scatteredptrs > $@

scattered-results.pdf: scattered-results.csv
	python plot_scattered_results.py scattered-results.csv

scaling-results-%.pdf: plot_scaling_results.py
	 python3 plot_scaling_results.py --optimizer $* --arsize 1024 --rsagsize 1073741824 --fusersagsize 67108864 --channels minimum --pdf $@
	
bandwidth-results.pdf: results-adam-16-gpus.csv results-adam-64-gpus.csv bandwidth-allreduce.py
	python3  bandwidth-allreduce.py $^

clean:
	rm -rf *.csv *.pdf ../*.pdf