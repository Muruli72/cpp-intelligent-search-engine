#include "TFIDF.h"

#include <cmath>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

using namespace std;


void TFIDF::addDocument(
    const string& filename,
    const vector<string>& tokens
) {

    totalDocuments++;

    documentWordCount[filename] = tokens.size();

    unordered_set<string> uniqueWords;

    for (const string& token : tokens) {

        // Count how many times this word appears
        // in this document.
        wordFrequency[filename][token]++;

        // Store the word only once.
        // We need this for document frequency.
        uniqueWords.insert(token);
    }

    // Count how many documents contain each word.
    for (const string& word : uniqueWords) {

        documentFrequency[word]++;
    }
}


double TFIDF::calculateTF(
    const string& filename,
    const string& word
) {

    // Check whether the document exists.
    if (documentWordCount.find(filename)
        == documentWordCount.end()) {

        return 0.0;
    }

    // Frequency of this word in the document.
    int frequency =
        wordFrequency[filename][word];

    // Total number of words in the document.
    int totalWords =
        documentWordCount[filename];

    if (totalWords == 0) {

        return 0.0;
    }

    // TF = word frequency / total words.
    return static_cast<double>(frequency)
           / totalWords;
}


double TFIDF::calculateIDF(
    const string& word
) {

    int documentsContainingWord =
        documentFrequency[word];

    if (documentsContainingWord == 0) {

        return 0.0;
    }

    // IDF = log(total documents / documents
    // containing this word).
    return log(
        static_cast<double>(totalDocuments)
        / documentsContainingWord
    );
}


double TFIDF::calculateTFIDF(
    const string& filename,
    const string& word
) {

    double tf =
        calculateTF(filename, word);

    double idf =
        calculateIDF(word);

    return tf * idf;
}


double TFIDF::calculateCosineSimilarity(
    const string& filename,
    const vector<string>& queryWords
) {

    if (queryWords.empty()) {

        return 0.0;
    }

    // ------------------------------------------
    // STEP 1: Count query word frequencies
    // ------------------------------------------

    unordered_map<string, int> queryFrequency;

    for (const string& word : queryWords) {

        queryFrequency[word]++;
    }

    // ------------------------------------------
    // STEP 2: Calculate dot product
    // ------------------------------------------

    double dotProduct = 0.0;

    for (const auto& entry : queryFrequency) {

        string word = entry.first;

        // TF of the word in the query.
        double queryTF =
            static_cast<double>(entry.second)
            / queryWords.size();

        // TF-IDF of the word in the query.
        double queryTFIDF =
            queryTF * calculateIDF(word);

        // TF-IDF of the same word in document.
        double documentTFIDF =
            calculateTFIDF(filename, word);

        // A[i] * B[i]
        dotProduct +=
            queryTFIDF * documentTFIDF;
    }

    // ------------------------------------------
    // STEP 3: Calculate query magnitude
    // ------------------------------------------

    double queryMagnitude = 0.0;

    for (const auto& entry : queryFrequency) {

        string word = entry.first;

        double queryTF =
            static_cast<double>(entry.second)
            / queryWords.size();

        double queryTFIDF =
            queryTF * calculateIDF(word);

        // square of the value
        queryMagnitude +=
            queryTFIDF * queryTFIDF;
    }

    queryMagnitude =
        sqrt(queryMagnitude);

    // ------------------------------------------
    // STEP 4: Calculate document magnitude
    // ------------------------------------------

    double documentMagnitude = 0.0;

    // Look at every word in this document.
    for (const auto& entry :
         wordFrequency[filename]) {

        string word = entry.first;

        double documentTFIDF =
            calculateTFIDF(filename, word);

        // square of the value
        documentMagnitude +=
            documentTFIDF * documentTFIDF;
    }

    documentMagnitude =
        sqrt(documentMagnitude);

    // ------------------------------------------
    // STEP 5: Avoid division by zero
    // ------------------------------------------

    if (queryMagnitude == 0.0 ||
        documentMagnitude == 0.0) {

        return 0.0;
    }

    // ------------------------------------------
    // STEP 6: Cosine similarity
    // ------------------------------------------

    return dotProduct /
           (queryMagnitude * documentMagnitude);
}


vector<pair<string, double>>
TFIDF::rankDocuments(
    const vector<string>& documents,
    const vector<string>& queryWords
) {

    vector<pair<string, double>> rankedResults;

    // Calculate cosine similarity for every
    // matching document.
    for (const string& document : documents) {

        double score =
            calculateCosineSimilarity(
                document,
                queryWords
            );

        rankedResults.push_back(
            {document, score}
        );
    }

    // Highest similarity first.
    sort(
        rankedResults.begin(),
        rankedResults.end(),
        [](const pair<string, double>& a,
           const pair<string, double>& b) {

            return a.second > b.second;
        }
    );

    return rankedResults;
}