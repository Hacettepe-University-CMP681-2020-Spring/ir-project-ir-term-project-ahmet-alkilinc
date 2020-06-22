import os

from pyrouge import Rouge155
import json
import time

start_time = time.time()

if __name__ == "__main__":
    # rouge_dir = '/home/ahmet/Downloads/pyrouge/rouge/tools/ROUGE-1.5.5/'
    # rouge_args = '-e /home/ahmet/Downloads/pyrouge/rouge/tools/ROUGE-1.5.5/data -n 4 -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -a -x -l 100'
    #
    # rouge = Rouge155(rouge_dir, rouge_args)
    #
    # # 'model' refers to the human summaries
    # rouge.model_dir = 'data/eval/'
    # rouge.model_filename_pattern = 'D3#ID#.M.100.T.[A-Z]'
    #
    # print( "-----------------MMR--------------------------")
    #
    # # 'system' or 'peer' refers to the system summaries
    # # We use the system summaries from 'ICSISumm' for an example
    # rouge.system_dir = 'Results/MMR_results/'
    # rouge.system_filename_pattern = 'd3(\d+)t.MMR'
    #
    # rouge_output = rouge.convert_and_evaluate()
    # output_dict = rouge.output_to_dict(rouge_output)
    #
    # print(json.dumps(output_dict, indent=2, sort_keys=True))
    print("-----------------MMR--------------------------")
    r = Rouge155()
    # set directories
    r.system_dir = 'Results/MMR_results/'
    r.model_dir = 'data/eval/'

    # define the patterns
    r.system_filename_pattern = 'd3(\d+)t.MMR'
    r.model_filename_pattern = 'D3#ID#.M.100.T.[A-Z]'

    # use default parameters to run the evaluation
    output_mmr = r.convert_and_evaluate()
    print(output_mmr)
    output_dict = r.output_to_dict(output_mmr)

    print("-----------------RBM--------------------------")

    # set directories
    r.system_dir = 'Results/RBM_results/'
    r.model_dir = 'data/eval/'

    # define the patterns
    r.system_filename_pattern = 'd3(\d+)t.RBM'
    r.model_filename_pattern = 'D3#ID#.M.100.T.[A-Z]'

    # use default parameters to run the evaluation
    output_rbm = r.convert_and_evaluate()
    print(output_rbm)
    output_dict = r.output_to_dict(output_rbm)

    print("-----------------LEXRANK--------------------------")

    # set directories
    r.system_dir = 'Results/LEXRANK_results/'
    r.model_dir = 'data/eval/'

    # define the patterns
    r.system_filename_pattern = 'd3(\d+)t.LEXRANK'
    r.model_filename_pattern = 'D3#ID#.M.100.T.[A-Z]'

    # use default parameters to run the evaluation
    output_lexrank = r.convert_and_evaluate()
    print(output_lexrank)
    output_dict = r.output_to_dict(output_lexrank)

    print("-----------------HYBRID--------------------------")

    # set directories
    r.system_dir = 'Results/HYBRID_results/'
    r.model_dir = 'data/eval/'

    # define the patterns
    r.system_filename_pattern = 'd3(\d+)t.HYBRID'
    r.model_filename_pattern = 'D3#ID#.M.100.T.[A-Z]'

    # use default parameters to run the evaluation
    output_hybrid = r.convert_and_evaluate()
    print(output_hybrid)
    output_dict = r.output_to_dict(output_hybrid)

    print("Execution time: " + str(time.time() - start_time))

    results = "*******************MMR*******************\n" + output_mmr + "\n*******************RBM" \
                                                                           "*******************\n" + output_rbm + \
              "\n*******************LEXRANK*******************\n" + output_lexrank +"\n*******************HYBRID" \
                                                                                    "*******************\n" + \
              output_hybrid

    results_folder = "Results/Rouge_result"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    filename = os.path.join(results_folder, ("Rouge_scores.txt"))
    with open(os.path.join(results_folder, ("Rouge_scores.txt")), "w") as fileOut:
        fileOut.write(results)
