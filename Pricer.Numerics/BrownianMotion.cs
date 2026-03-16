namespace Pricer.Numerics;

/// <summary>
/// Represents a single point in a Brownian motion path with a timestamp.
/// </summary>
public class BrownianMotionPoint
{
    public DateTime Time { get; set; }
    public double Value { get; set; }

    public BrownianMotionPoint(DateTime time, double value)
    {
        Time = time;
        Value = value;
    }
}

/// <summary>
/// Generates Brownian motion paths and increments.
/// </summary>
public class BrownianMotionGenerator
{
    private readonly Random _rng;

    public BrownianMotionGenerator(Random? rng = null)
    {
        _rng = rng ?? new Random();
    }

    /// <summary>
    /// Generates a single Brownian motion path.
    /// </summary>
    /// <param name="timeHorizon">Time horizon in years</param>
    /// <param name="steps">Number of time steps</param>
    /// <param name="drift">Drift coefficient (μ)</param>
    /// <param name="volatility">Volatility coefficient (σ)</param>
    /// <param name="scalingExponent">Exponent for volatility scaling (α) in σ·(Δt)^α</param>
    /// <param name="startDate">Starting date for the path</param>
    /// <returns>List of Brownian motion points</returns>
    public List<BrownianMotionPoint> GeneratePath(
        double timeHorizon,
        int steps,
        double drift,
        double volatility,
        double scalingExponent,
        DateTime? startDate = null)
    {
        startDate ??= new DateTime(DateTime.Now.Year, 1, 1);
        
        double dt = timeHorizon / steps;
        double scaledVolatility = volatility * Math.Pow(dt, scalingExponent);

        var path = new List<BrownianMotionPoint>(steps + 1);
        double w = 0.0;

        for (int i = 0; i <= steps; i++)
        {
            double daysFromStart = (i * dt) * 365.0;
            path.Add(new BrownianMotionPoint(startDate.Value.AddDays(daysFromStart), w));

            if (i < steps)
            {
                w += drift * dt + scaledVolatility * SampleStandardNormal();
            }
        }

        return path;
    }

    /// <summary>
    /// Generates multiple Brownian motion paths.
    /// </summary>
    public List<List<BrownianMotionPoint>> GeneratePaths(
        double timeHorizon,
        int steps,
        double drift,
        double volatility,
        double scalingExponent,
        int numPaths,
        DateTime? startDate = null)
    {
        var paths = new List<List<BrownianMotionPoint>>(numPaths);
        for (int p = 0; p < numPaths; p++)
        {
            paths.Add(GeneratePath(timeHorizon, steps, drift, volatility, scalingExponent, startDate));
        }
        return paths;
    }

    /// <summary>
    /// Generates two correlated Brownian motion paths.
    /// </summary>
    /// <param name="timeHorizon">Time horizon in years</param>
    /// <param name="steps">Number of time steps</param>
    /// <param name="drift">Drift coefficient (μ)</param>
    /// <param name="volatility">Volatility coefficient (σ)</param>
    /// <param name="correlationCoefficient">Correlation coefficient (ρ) between -1 and 1</param>
    /// <param name="startDate">Starting date for the paths</param>
    /// <returns>Tuple of (path1, path2)</returns>
    public (List<BrownianMotionPoint> Path1, List<BrownianMotionPoint> Path2) GenerateCorrelatedPaths(
        double timeHorizon,
        int steps,
        double drift,
        double volatility,
        double correlationCoefficient,
        DateTime? startDate = null)
    {
        startDate ??= new DateTime(DateTime.Now.Year, 1, 1);
        
        // Ensure correlation coefficient is valid
        double rho = Math.Clamp(correlationCoefficient, -1.0, 1.0);
        double sqrtOneMinusRho2 = Math.Sqrt(1.0 - rho * rho);

        double dt = timeHorizon / steps;
        double sqrtDt = Math.Sqrt(dt);

        var path1 = new List<BrownianMotionPoint>(steps + 1);
        var path2 = new List<BrownianMotionPoint>(steps + 1);

        double w1 = 0.0;
        double w2 = 0.0;

        for (int i = 0; i <= steps; i++)
        {
            double daysFromStart = (i * dt) * 365.0;
            path1.Add(new BrownianMotionPoint(startDate.Value.AddDays(daysFromStart), w1));
            path2.Add(new BrownianMotionPoint(startDate.Value.AddDays(daysFromStart), w2));

            if (i < steps)
            {
                // Generate two independent standard normal random variables
                double z1 = SampleStandardNormal();
                double z2 = SampleStandardNormal();

                // Create correlated increments using Cholesky decomposition
                // w1 uses z1 directly
                // w2 uses a linear combination: ρ*z1 + sqrt(1-ρ²)*z2
                double dw1 = drift * dt + volatility * sqrtDt * z1;
                double dw2 = drift * dt + volatility * sqrtDt * (rho * z1 + sqrtOneMinusRho2 * z2);

                w1 += dw1;
                w2 += dw2;
            }
        }

        return (path1, path2);
    }

    /// <summary>
    /// Samples from a standard normal distribution using the Box-Muller transform.
    /// </summary>
    private double SampleStandardNormal()
    {
        double u1 = 1.0 - _rng.NextDouble();
        double u2 = 1.0 - _rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
