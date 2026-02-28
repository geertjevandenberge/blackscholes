using MathNet.Numerics.Distributions;


namespace Pricer.Numerics;

public enum OptionType
{
    Call,
    Put
}

public class BlackScholes
{
    private OptionType option;
    private double r; // Risk-free interest rate
    private double T; // Time to maturity
    private double sigma; // Volatility
    private double K; // Strike price
    private double S; // Underlying asset price
    private double q; // Dividend yield

    public BlackScholes(
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

    // Cumulative normal distribution
    private static double N(double x)
        => Normal.CDF(0.0, 1.0, x);

    // Normal probability density function
    private static double n(double x)
        => Math.Exp(-0.5 * x * x) / Math.Sqrt(2.0 * Math.PI);


    // Black–Scholes price
    public double Price()
    {
        // Handle immediate expiry
        if (T <= 0.0)
        {
            return option == OptionType.Call
                ? Math.Max(S - K, 0.0)
                : Math.Max(K - S, 0.0);
        }

        // If volatility is zero or extremely small, option is essentially intrinsic (discounted forward)
        if (sigma <= 0.0)
        {
            var forwardSpot = S * Math.Exp(-q * T);
            var discountedStrike = K * Math.Exp(-r * T);
            return option == OptionType.Call
                ? Math.Max(forwardSpot - discountedStrike, 0.0)
                : Math.Max(discountedStrike - forwardSpot, 0.0);
        }

        var sqrtT = Math.Sqrt(T);
        var d1 = (Math.Log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
        var d2 = d1 - sigma * sqrtT;

        if (option == OptionType.Call)
        {
            return S * Math.Exp(-q * T) * N(d1) - K * Math.Exp(-r * T) * N(d2);
        }
        else
        {
            return K * Math.Exp(-r * T) * N(-d2) - S * Math.Exp(-q * T) * N(-d1);
        }
    }

    public double Vega()
    {
        if (T <= 0.0 || sigma <= 0.0)
            return 0.0;

        var sqrtT = Math.Sqrt(T);
        var d1 = (Math.Log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
        return S * Math.Exp(-q * T) * n(d1) * sqrtT;
    }

}

// Test price compared with party at the mooonlight option calculator
// Test put call parity for consistency


public static class ImpliedVolatilityCalculator
{
    public static double Compute(
        OptionType optionType,
        double marketPrice,
        double interestRate,
        double timeToMaturity,
        double strike,
        double underlyingPrice,
        double initialGuess = 0.2,
        double tolerance = 1e-8,
        int maxIterations = 100)
    {
        if (marketPrice < 0.0)
            throw new ArgumentException("Market price must be non-negative", nameof(marketPrice));

        double sigma = initialGuess > 0.0 ? initialGuess : 0.2;

        for (int i = 0; i < maxIterations; ++i)
        {
            var bs = new BlackScholes(optionType, interestRate, timeToMaturity, sigma, strike, underlyingPrice);
            double modelPrice = bs.Price();
            double diff = modelPrice - marketPrice;

            if (Math.Abs(diff) < tolerance)
                return sigma;

            double vega = bs.Vega();

            // If Vega is too small, Newton step may blow up; bail out
            if (vega <= 1e-12)
                break;

            sigma -= diff / vega;

            // keep sigma in a reasonable positive range
            if (sigma <= 0)
                sigma = 1e-12;
        }

        throw new InvalidOperationException("Implied volatility did not converge");
    }

    public static double BachelierImpliedVolATM(
        double optionPrice,
        double underlyingPrice,
        double timeToMaturity,
        double interestRate)
    {
       if (optionPrice < 0.0)
           throw new ArgumentException("Option price must be non-negative", nameof(optionPrice));

       if (timeToMaturity <= 0.0)
           throw new ArgumentException("Time to maturity must be positive", nameof(timeToMaturity));

       // For ATM-forward Bachelier model the undiscounted ATM price is: C = e^{-rT} * sigma_N * sqrt(T) / sqrt(2π)
       // Solve for sigma_N: sigma_N = C * e^{rT} * sqrt(2π) / sqrt(T)
       var sigmaN = optionPrice * Math.Exp(interestRate * timeToMaturity) * Math.Sqrt(2.0 * Math.PI) / Math.Sqrt(timeToMaturity);
       return sigmaN;
    }
}