import pytrec_eval
import numpy as np
import csv
import json
import matplotlib.pyplot as plt
def get_vals(filepath):
    results = {}
    with open(filepath, mode='r', newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        for line in reader:
            topicID = line[0]
            answerID = line[2]
            relevance = float(line[4])

            if topicID in results.keys():
                results[topicID][answerID] = relevance
            else:
                results[topicID] = {answerID: relevance}

    qrel = {}
    with open('qrel_1.tsv', mode='r', newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        for line in reader:
            topicID = line[0]
            answerID = line[2]
            relevance = int(line[3])
            if topicID in qrel.keys():
                qrel[topicID][answerID] = relevance
            else:
                qrel[topicID] = {answerID: relevance}

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'ndcg_cut_5','ndcg_cut_10','ndcg', 'P_5', 'P_10', 'P', 'map', 'bpref'})

    # Evaluate the results
    eval = evaluator.evaluate(results)

    # Prepare a dictionary to store results for every query
    query_results = {}

    # Collect results for each topic
    for topic in eval:
        query_results[topic] = {
            "nDCG@5": eval[topic].get("ndcg_cut_5", 0.0),
            "nDCG@10": eval[topic].get("ndcg_cut_10", 0.0),
            "nDCG@All": eval[topic].get("ndcg_cut_5", 0.0),
            "P_5": eval[topic].get("P_5", 0.0),        
            "P_10": eval[topic].get("P_10", 0.0),
            "P_All": eval[topic].get("P", 0.0),
            "MAP": eval[topic].get("map", 0.0),
            "bpref": eval[topic].get("bpref", 0.0),
            
        }

    # Save the query results to a JSON file
    with open('graph_eval_store.json', 'w') as json_file:
        json.dump(query_results, json_file, indent=4)

    # Calculate averages
    num_topics = len(eval)
    average_results = {
        "Average nDCG@5": sum(res["nDCG@5"] for res in query_results.values()) / num_topics,
        "Average nDCG@10": sum(res["nDCG@10"] for res in query_results.values()) / num_topics,
        "Average nDCG@All": sum(res["nDCG@All"] for res in query_results.values()) / num_topics,
        "Average P_5": sum(res["P_5"] for res in query_results.values()) / num_topics,
        "Average P_10": sum(res["P_10"] for res in query_results.values()) / num_topics,
        "Average P_All": sum(res["P_All"] for res in query_results.values()) / num_topics,
        "Average MAP": sum(res["MAP"] for res in query_results.values()) / num_topics,
        "Average Bpref": sum(res["P_10"] for res in query_results.values()) / num_topics,
    }

    # Prepare data for P@5 plot
    P_5_Topic_ID = {topic: eval[topic]["P_5"] for topic in eval}

    # Sort topics by P@5 values
    P_5_values_sorted = sorted(P_5_Topic_ID, key=lambda x: P_5_Topic_ID[x], reverse=True)

    # Prepare values for plotting
    vals = [P_5_Topic_ID[id] for id in P_5_values_sorted]
    return vals

def get_vals_space(filepath):
    results = {}
    with open(filepath, mode='r', newline='') as file:
        reader = csv.reader(file, delimiter=' ')
        for line in reader:
            topicID = line[0]
            answerID = line[2]
            relevance = float(line[4])

            if topicID in results.keys():
                results[topicID][answerID] = relevance
            else:
                results[topicID] = {answerID: relevance}

    qrel = {}
    with open('qrel_1.tsv', mode='r', newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        for line in reader:
            topicID = line[0]
            answerID = line[2]
            relevance = int(line[3])
            if topicID in qrel.keys():
                qrel[topicID][answerID] = relevance
            else:
                qrel[topicID] = {answerID: relevance}

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'ndcg_cut_5','ndcg_cut_10','ndcg', 'P_5', 'P_10', 'P', 'map', 'bpref'})

    # Evaluate the results
    eval = evaluator.evaluate(results)

    # Prepare a dictionary to store results for every query
    query_results = {}

    # Collect results for each topic
    for topic in eval:
        query_results[topic] = {
            "nDCG@5": eval[topic].get("ndcg_cut_5", 0.0),
            "nDCG@10": eval[topic].get("ndcg_cut_10", 0.0),
            "nDCG@All": eval[topic].get("ndcg_cut_5", 0.0),
            "P_5": eval[topic].get("P_5", 0.0),        
            "P_10": eval[topic].get("P_10", 0.0),
            "P_All": eval[topic].get("P", 0.0),
            "MAP": eval[topic].get("map", 0.0),
            "bpref": eval[topic].get("bpref", 0.0),
            
        }

    # Save the query results to a JSON file
    with open('graph_eval_store.json', 'w') as json_file:
        json.dump(query_results, json_file, indent=4)

    # Calculate averages
    num_topics = len(eval)
    average_results = {
        "Average nDCG@5": sum(res["nDCG@5"] for res in query_results.values()) / num_topics,
        "Average nDCG@10": sum(res["nDCG@10"] for res in query_results.values()) / num_topics,
        "Average nDCG@All": sum(res["nDCG@All"] for res in query_results.values()) / num_topics,
        "Average P_5": sum(res["P_5"] for res in query_results.values()) / num_topics,
        "Average P_10": sum(res["P_10"] for res in query_results.values()) / num_topics,
        "Average P_All": sum(res["P_All"] for res in query_results.values()) / num_topics,
        "Average MAP": sum(res["MAP"] for res in query_results.values()) / num_topics,
        "Average Bpref": sum(res["P_10"] for res in query_results.values()) / num_topics,
    }

    # Prepare data for P@5 plot
    P_5_Topic_ID = {topic: eval[topic]["P_5"] for topic in eval}

    # Sort topics by P@5 values
    P_5_values_sorted = sorted(P_5_Topic_ID, key=lambda x: P_5_Topic_ID[x], reverse=True)

    # Prepare values for plotting
    vals = [P_5_Topic_ID[id] for id in P_5_values_sorted]
    return vals

# vals10 = get_vals("ft_cross_results\\result_ce_ft10_1.tsv")[:100]
# vals20 = get_vals("ft_cross_results\\result_ce_ft20_1.tsv")[:100]
# vals30 = get_vals("ft_cross_results\\result_ce_ft30_1.tsv")[:100]
# vals40 = get_vals("ft_cross_results\\result_ce_ft40_1.tsv")[:100]
# vals50 = get_vals("ft_cross_results\\result_ce_ft50_1.tsv")[:100]
vals_bi_og = get_vals("result_bi_1.tsv")
vals_cross_og = get_vals("result_ce_1.tsv")
vals_bm25 = get_vals_space("BM25_results_1.tsv")
vals_tfidf = get_vals_space("TFIDF_results_1.tsv")


# Plot P@5 values
plt.figure(figsize=(10, 6))
# plt.plot(range(len(vals10)), vals10, linestyle='-', color='red',label="10 epochs")
# plt.plot(range(len(vals20)), vals20, linestyle='-', color='orange',label="20 epochs")
# plt.plot(range(len(vals30)), vals30, linestyle='-', color='blue',label="30 epochs")
# plt.plot(range(len(vals40)), vals40, linestyle='-', color='green',label="40 epochs")
# plt.plot(range(len(vals50)), vals50, linestyle='-', color='purple',label="50 epochs")

plt.plot(range(len(vals_bi_og)), vals_bi_og, linestyle='-', color='red',label="Bi-Encoder")
plt.plot(range(len(vals_cross_og)), vals_cross_og, linestyle='-', color='orange',label="Cross-Encoder")
plt.plot(range(len(vals_bm25)), vals_bm25, linestyle='-', color='blue',label="BM25")
plt.plot(range(len(vals_tfidf)), vals_tfidf, linestyle='-', color='purple',label="TF-IDF")





# Label the axes
plt.xlabel('Doc Rank')
plt.ylabel('P@5')
plt.title('P@5 Untrained Encoders VS. Old Methods (topics_1)')

plt.xticks(fontsize=10)
xticks = plt.gca().get_xticks()
plt.xticks(xticks)

# Show the plot
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
