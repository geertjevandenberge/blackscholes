using MathNet.Numerics.Distributions;

namespace Pricer.Numerics;

public sealed class BlackScholesGreeks
{
    private readonly OptionType option;
    private readonly double r;
    private readonly double T;
    private readonly double sigma;
    private readonly double K;
    private readonly double S;
    private readonly double q;

    public BlackScholesGreeks(
        OptionType optionType,
        double riskFreeRate,
        double timeToMaturity,
        double volatility,
        double strike,
        double underlyingPrice,
        double dividendYield = 0.0)
    {
        option = optionType;
        r = riskFreeRate;
        T = timeToMaturity;
        sigma = volatility;
        K = strike;
        S = underlyingPrice;
        q = dividendYield;
    }

    private static double N(double x) => Normal.CDF(0.0, 1.0, x);

    private static double n(double x) => Math.Exp(-0.5 * x * x) / Math.Sqrt(2.0 * Math.PI);

    private (double d1, double d2, double sqrtT) GetD1D2()
    {
        var sqrtT = Math.Sqrt(T);
        var d1 = (Math.Log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
        var d2 = d1 - sigma * sqrtT;
        return (d1, d2, sqrtT);
    }

    public double Price()
    {
        if (T <= 0.0)
        {
            return option == OptionType.Call
                ? Math.Max(S - K, 0.0)
                : Math.Max(K - S, 0.0);
        }

        if (sigma <= 0.0)
        {
            var forwardSpot = S * Math.Exp(-q * T);
            var discountedStrike = K * Math.Exp(-r * T);
            return option == OptionType.Call
                ? Math.Max(forwardSpot - discountedStrike, 0.0)
                : Math.Max(discountedStrike - forwardSpot, 0.0);
        }

        var (d1, d2, _) = GetD1D2();

        return option == OptionType.Call
            ? S * Math.Exp(-q * T) * N(d1) - K * Math.Exp(-r * T) * N(d2)
            : K * Math.Exp(-r * T) * N(-d2) - S * Math.Exp(-q * T) * N(-d1);
    }

    public double Delta()
    {
        if (T <= 0.0 || sigma <= 0.0)
            return 0.0;

        var (d1, _, _) = GetD1D2();

        return option == OptionType.Call
            ? Math.Exp(-q * T) * N(d1)
            : -Math.Exp(-q * T) * N(-d1);
    }

    public double Gamma()
    {
        if (T <= 0.0 || sigma <= 0.0)
            return 0.0;

        var (d1, _, sqrtT) = GetD1D2();

        return Math.Exp(-q * T) * n(d1) / (S * sigma * sqrtT);
    }

    public double Theta()
    {
        if (T <= 0.0 || sigma <= 0.0)
            return 0.0;

        var (d1, d2, sqrtT) = GetD1D2();

        var firstTerm = -S * Math.Exp(-q * T) * n(d1) * sigma / (2.0 * sqrtT);

        if (option == OptionType.Call)
        {
            return firstTerm
                   - r * K * Math.Exp(-r * T) * N(d2)
                   + q * S * Math.Exp(-q * T) * N(d1);
        }

        return firstTerm
               + r * K * Math.Exp(-r * T) * N(-d2)
               - q * S * Math.Exp(-q * T) * N(-d1);
    }

    public double Vega()
    {
        if (T <= 0.0 || sigma <= 0.0)
            return 0.0;

        var (d1, _, sqrtT) = GetD1D2();

        return S * Math.Exp(-q * T) * n(d1) * sqrtT;
    }
}
