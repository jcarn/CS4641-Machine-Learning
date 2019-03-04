package opt.test;

import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * A test of the knap sack problem
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class KnapsackTest {
    /** Random number generator */
    private static final Random random = new Random();
    /** The number of items */
    private static final int NUM_ITEMS = 40;
    /** The number of copies each */
    private static final int COPIES_EACH = 4;
    /** The maximum weight for a single element */
    private static final double MAX_WEIGHT = 50;
    /** The maximum volume for a single element */
    private static final double MAX_VOLUME = 50;
    /** The volume of the knapsack */
    private static final double KNAPSACK_VOLUME = 
         MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4;

    private static DecimalFormat def = new DecimalFormat("0.000");

    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        int[] copies = new int[NUM_ITEMS];
        Arrays.fill(copies, COPIES_EACH);
        double[] weights = new double[NUM_ITEMS];
        double[] volumes = new double[NUM_ITEMS];
        for (int i = 0; i < NUM_ITEMS; i++) {
            weights[i] = random.nextDouble() * MAX_WEIGHT;
            volumes[i] = random.nextDouble() * MAX_VOLUME;
        }
         int[] ranges = new int[NUM_ITEMS];
        Arrays.fill(ranges, COPIES_EACH + 1);
        EvaluationFunction ef = new KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        long starttime = System.currentTimeMillis();
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
        FixedIterationTrainer fit = null;
        System.out.println("Randomized Hill Climbing");
        for (int i = 500; i <= 20000; i += 500) {
            System.out.println("----------" + i + " iterations-----------");
            fit = new FixedIterationTrainer(rhc, i);
            fit.train();
            System.out.println("RHC: " + ef.value(rhc.getOptimal()));
            System.out.println("Time : " + def.format((System.currentTimeMillis() - starttime) / Math.pow(10, 3)) + " seconds");
        }
        System.out.println("============================");


        // Simulated Annealing
        starttime = System.currentTimeMillis();
        SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .8, hcp);
        System.out.println("Simulated Annealing");
        for (int i = 500; i <= 20000; i += 500) {
            System.out.println("----------" + i + " iterations-----------");
            fit = new FixedIterationTrainer(sa, i);
            fit.train();
            System.out.println("SA: " + ef.value(sa.getOptimal()));
            System.out.println("Time : " + def.format((System.currentTimeMillis() - starttime) / Math.pow(10, 3)) + " seconds");
        }
        System.out.println("============================");

        // Genetic Algorithm
        starttime = System.currentTimeMillis();
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 10, 60, gap);
        System.out.println("Genetic Algorithm");
        for (int i = 500; i <= 20000; i += 500) {
            System.out.println("----------" + i + " iterations-----------");
            fit = new FixedIterationTrainer(ga, i);
            fit.train();
            System.out.println("GA: " + ef.value(ga.getOptimal()));
            System.out.println("Time : " + def.format((System.currentTimeMillis() - starttime) / Math.pow(10, 3)) + " seconds");
        }
        System.out.println("============================");

        starttime = System.currentTimeMillis();
        MIMIC mimic = new MIMIC(00, 100, pop);
        System.out.println("MIMIC");
        for (int i = 500; i <= 20000; i += 500) {
            System.out.println("----------" + i + " iterations-----------");
            fit = new FixedIterationTrainer(mimic, i);
            fit.train();
            System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));
            System.out.println("Time : " + def.format((System.currentTimeMillis() - starttime) / Math.pow(10, 3)) + " seconds");
        }
    }

}
