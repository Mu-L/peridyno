#include <iostream>
#include <string>
#include <vector>
#include <cctype>
#include <algorithm>
#include <sstream>

namespace DescriptionHelper 
{
    bool iequals(const std::string& a, const std::string& b) {
        return std::equal(a.begin(), a.end(), b.begin(), b.end(),
            [](char c1, char c2) { return std::tolower(c1) == std::tolower(c2); });
    }

    std::string trim(const std::string& s) {
        size_t start = s.find_first_not_of(" \t\n\r\f\v");
        if (start == std::string::npos) return "";
        size_t end = s.find_last_not_of(" \t\n\rf\v");
        return s.substr(start, end - start + 1);
    }

    std::string collapseSpaces(const std::string& s) {
        std::string result;
        bool inSpace = false;
        for (char ch : s) {
            if (std::isspace(ch)) {
                if (!inSpace && !result.empty()) {
                    result += ' ';
                    inSpace = true;
                }
            }
            else {
                result += ch;
                inSpace = false;
            }
        }
        return trim(result);
    }

    bool parseQtStyleDescriptionRobust(const std::string& input,
        std::string& cleanDescription,
        bool& IsVLayout,
        bool& onlyDetail) {
        IsVLayout = true;
        onlyDetail = false;
        std::string working = input;
        bool foundAny = false;

        auto findQtStylePos = [](const std::string& str, size_t start = 0) -> size_t {
            const std::string marker = "qtstyle";
            size_t pos = start;
            while (pos < str.length()) {
                size_t found = str.find_first_of("Qq", pos);
                if (found == std::string::npos) break;
                if (str.length() - found < marker.length()) break;
                if (iequals(str.substr(found, marker.length()), marker)) {
                    size_t after = found + marker.length();
                    while (after < str.length() && std::isspace(str[after])) ++after;
                    if (after < str.length() && str[after] == '(') {
                        return found;
                    }
                    else {
                        pos = found + 1;
                        continue;
                    }
                }
                pos = found + 1;
            }
            return std::string::npos;
        };

        while (true) {
            size_t startPos = findQtStylePos(working);
            if (startPos == std::string::npos) break;

            size_t nameEnd = startPos + std::string("qtstyle").length();
            size_t parenPos = nameEnd;
            while (parenPos < working.length() && std::isspace(working[parenPos])) ++parenPos;
            if (parenPos >= working.length() || working[parenPos] != '(') break;
            size_t openParen = parenPos;

            int depth = 1;
            size_t closeParen = openParen + 1;
            while (closeParen < working.length() && depth > 0) {
                if (working[closeParen] == '(') depth++;
                else if (working[closeParen] == ')') depth--;
                if (depth == 0) break;
                ++closeParen;
            }
            if (depth != 0) break;

            std::string params = working.substr(openParen + 1, closeParen - openParen - 1);
            std::vector<std::string> tokens;
            std::stringstream ss(params);
            std::string token;
            while (std::getline(ss, token, ',')) {
                std::string trimmed = trim(token);
                if (!trimmed.empty()) tokens.push_back(trimmed);
            }

            for (const auto& t : tokens) {
                if (iequals(t, "VLayout")) {
                    IsVLayout = true;
                }
                else if (iequals(t, "HLayout")) {
                    IsVLayout = false;
                }
                else if (iequals(t, "OnlyDetail")) {
                    onlyDetail = true;
                }
            }

            working.erase(startPos, closeParen - startPos + 1);
            foundAny = true;
        }

        cleanDescription = collapseSpaces(working);
        return foundAny;
    }

    std::string parseQtStyleGroup(const std::string& input) {
        size_t pos = 0;
        while (pos < input.length()) {
            size_t found = input.find_first_of("Qq", pos);
            if (found == std::string::npos) break;
            
            const std::string marker = "QtStyle";
            if (input.length() - found < marker.length()) {
                pos = found + 1;
                continue;
            }
            
            bool match = true;
            for (size_t i = 0; i < marker.length(); i++) {
                if (std::tolower(input[found + i]) != std::tolower(marker[i])) {
                    match = false;
                    break;
                }
            }
            if (!match) {
                pos = found + 1;
                continue;
            }
            
            size_t after = found + marker.length();
            while (after < input.length() && std::isspace(input[after])) ++after;
            if (after >= input.length() || input[after] != '(') {
                pos = found + 1;
                continue;
            }
            
            size_t openParen = after;
            int depth = 1;
            size_t closeParen = openParen + 1;
            while (closeParen < input.length() && depth > 0) {
                if (input[closeParen] == '(') depth++;
                else if (input[closeParen] == ')') depth--;
                if (depth == 0) break;
                ++closeParen;
            }
            if (depth != 0) {
                pos = found + 1;
                continue;
            }
            
            std::string params = input.substr(openParen + 1, closeParen - openParen - 1);
            std::vector<std::string> tokens;
            std::stringstream ss(params);
            std::string token;
            while (std::getline(ss, token, ',')) {
                std::string trimmed = trim(token);
                if (!trimmed.empty()) tokens.push_back(trimmed);
            }
            
            std::string firstPosArg;
            for (const auto& t : tokens) {
                size_t eqPos = t.find('=');
                if (eqPos != std::string::npos) {
                    std::string key = trim(t.substr(0, eqPos));
                    std::string value = trim(t.substr(eqPos + 1));
                    if (iequals(key, "Group") && !value.empty()) {
                        return value;
                    }
                } else {
                    if (firstPosArg.empty())
                        firstPosArg = t;
                }
            }
            
            if (!firstPosArg.empty())
                return firstPosArg;
            
            pos = closeParen + 1;
        }
        
        return "";
    }

}
