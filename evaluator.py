import pytrec_eval
import numpy as np
import csv
import json
import matplotlib.pyplot as plt

results = {}
with open('result_ce_1.tsv', mode='r', newline='') as file:
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
with open('evaluation_result_ce_1.json', 'w') as json_file:
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


# Print the average results
print(f"Average nDCG@5: {average_results['Average nDCG@5']}")
print(f"Average nDCG@10: {average_results['Average nDCG@10']}")
print(f"Average nDCG@All: {average_results['Average nDCG@All']}")
print(f"Average P_5: {average_results['Average P_5']}")
print(f"Average P_10: {average_results['Average P_10']}")
print(f"Average P_All: {average_results['Average P_All']}")
print(f"Average MAP: {average_results['Average MAP']}")
print(f"Average Bpref: {average_results['Average Bpref']}")

# Prepare data for P@5 plot
P_5_Topic_ID = {topic: eval[topic]["P_5"] for topic in eval}

# Sort topics by P@5 values
P_5_values_sorted = sorted(P_5_Topic_ID, key=lambda x: P_5_Topic_ID[x], reverse=True)

# Prepare values for plotting
vals = [P_5_Topic_ID[id] for id in P_5_values_sorted]

# Plot P@5 values
plt.figure(figsize=(10, 6))
plt.plot(P_5_values_sorted, vals, marker='o', linestyle='-', color='b')

# Label the axes
plt.xlabel('DocID')
plt.ylabel('P@5')
plt.title('P@5 for Untrained Cross-Encoder (topics_1)')

plt.xticks(rotation=60, fontsize=5)
xticks = plt.gca().get_xticks()
plt.xticks(xticks[::100])  # Show every 10th tick

# Show the plot
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
