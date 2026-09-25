#ifndef TFIDF_H
#define TFIDF_H

#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

class TFIDF {

private:

    // document -> word -> frequency
    std::unordered_map<
        std::string,
        std::unordered_map<std::string, int>
    > wordFrequency;

    // document -> total number of words
    std::unordered_map<std::string, int> documentWordCount;

    // word -> number of documents containing the word
    std::unordered_map<std::string, int> documentFrequency;

    // Total number of documents
    int totalDocuments = 0;

public:

    void addDocument(
        const std::string& filename,
        const std::vector<std::string>& tokens
    );

    double calculateTF(
        const std::string& filename,
        const std::string& word
    );

    double calculateIDF(
        const std::string& word
    );

    double calculateTFIDF(
        const std::string& filename,
        const std::string& word
    );

    double calculateCosineSimilarity(
        const std::string& filename,
        const std::vector<std::string>& queryWords
    );

    std::vector<std::pair<std::string, double>> rankDocuments(
        const std::vector<std::string>& documents,
        const std::vector<std::string>& queryWords
    );
};

#endif
