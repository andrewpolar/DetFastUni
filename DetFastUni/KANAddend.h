#pragma once
#include "Urysohn.h"
#include "Univariate.h"
#include <memory>

class KANAddend
{
public:
    KANAddend(const std::vector<double>& xmin, const std::vector<double>& xmax,
        double targetMin, double targetMax,
        int inner, int outer, int number_of_inputs) {
        _lastInnerValue = 0.0;
        _u = std::make_unique<Urysohn>(xmin, xmax, targetMin, targetMax, inner, number_of_inputs);
        _univariate = std::make_unique<Univariate>(targetMin, targetMax, targetMin, targetMax, outer);
    }
    KANAddend(const KANAddend& addend) {
        _lastInnerValue = addend._lastInnerValue;
        _univariate = std::make_unique<Univariate>(*addend._univariate);
        _u = std::make_unique<Urysohn>(*addend._u);
    }
    void UpdateUsingMemory(double diff) {
        double derrivative = _univariate->GetDerivative(_lastInnerValue);
        _u->UpdateUsingMemory(diff * derrivative);
        _univariate->UpdateUsingMemory(diff);
    }
    void UpdateUsingInput(const std::vector<double>& input, double diff) {
        double value = _u->GetValueUsingInput(input);
        double derrivative = _univariate->GetDerivative(value);
        _u->UpdateUsingInput(diff * derrivative, input);
        _univariate->UpdateUsingInput(value, diff);
    }
    double ComputeUsingInput(const std::vector<double>& input, bool noUpdate = false) {
        _lastInnerValue = _u->GetValueUsingInput(input, noUpdate);
        return _univariate->GetFunctionUsingInput(_lastInnerValue, noUpdate);
    }
    void IncrementInner() {
        _u->IncrementInner();
    }
    void IncrementOuter() {
        _univariate->IncrementPoints();
    }
    int HowManyOuter() {
        return _univariate->HowManyPoints();
    }
    int HowManyInner() {
        return _u->_univariateList[0]->HowManyPoints();
    }
    std::vector<double> GetAllOuterPoints() {
        return _univariate->GetAllPoints();
    }
    double GetLastDerivative() {
        return _univariate->GetDerivative(_lastInnerValue);
    }
    void RescaleAddend(const KANAddend& a, int n) {
        _u->RescaleUrysohn(*a._u, n);
        _univariate->RescaleFunction(a._univariate->_y, n);
    }
    double _lastInnerValue;
    std::unique_ptr<Urysohn> _u;
    std::unique_ptr<Univariate> _univariate;
};
