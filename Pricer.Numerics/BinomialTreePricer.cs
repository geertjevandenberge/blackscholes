namespace Pricer.Numerics;

/// <summary>
/// Cox-Ross-Rubinstein (CRR) binomial tree pricer for European options.
/// </summary>
public sealed class BinomialTreePricer
{
    private readonly OptionType _optionType;
    private readonly double _r;
    private readonly double _T;
    private readonly double _sigma;
    private readonly double _K;
    private readonly double _S;
    private readonly double _q;

    public BinomialTreePricer(
        OptionType optionType,
        double riskFreeRate,
        double timeToMaturity,
        double volatility,
        double strike,
        double underlyingPrice,
        double dividendYield = 0.0)
    {
        _optionType = optionType;
        _r = riskFreeRate;
        _T = timeToMaturity;
        _sigma = volatility;
        _K = strike;
        _S = underlyingPrice;
        _q = dividendYield;
    }

    /// <summary>
    /// Prices the option using an n-step CRR binomial tree.
    /// </summary>
    public double Price(int n)
    {
        if (n <= 0)
            throw new ArgumentOutOfRangeException(nameof(n), "Number of steps must be positive.");

        double dt = _T / n;
        double u = Math.Exp(_sigma * Math.Sqrt(dt));
        double d = 1.0 / u;
        double discount = Math.Exp(-_r * dt);
        double pUp = (Math.Exp((_r - _q) * dt) - d) / (u - d);
        double pDown = 1.0 - pUp;

        // Terminal payoffs
        var values = new double[n + 1];
        for (int j = 0; j <= n; j++)
        {
            double spotAtNode = _S * Math.Pow(u, j) * Math.Pow(d, n - j);
            values[j] = _optionType == OptionType.Call
                ? Math.Max(spotAtNode - _K, 0.0)
                : Math.Max(_K - spotAtNode, 0.0);
        }

        // Backward induction
        for (int i = n - 1; i >= 0; i--)
        {
            for (int j = 0; j <= i; j++)
            {
                values[j] = discount * (pUp * values[j + 1] + pDown * values[j]);
            }
        }

        return values[0];
    }
}
