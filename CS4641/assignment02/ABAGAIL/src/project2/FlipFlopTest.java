package project2;

import java.text.DecimalFormat;
import java.util.Arrays;

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
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import shared.Instance;

import java.io.*;
import java.util.Random;

/**
 * A test using the Flip Flop evaluation function
 * @author James Liu
 * @version 1.0
 */
public class FlipFlopTest {
    private static DecimalFormat def = new DecimalFormat("0.000");

    public static void main(String[] args) throws IOException {
        int N = 5000;

        Random random = new Random();

        File train_file = new File("src/results/" + "flipflop_rhc" + "_train.csv");
        BufferedWriter trainWriter;
        trainWriter = new BufferedWriter(new FileWriter(train_file));

        int iterations = 20;
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);

        double[][] points = new double[N][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();
        }
//        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);

        EvaluationFunction ef = new FlipFlopEvaluationFunction();
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        int per_iter = 5;

       long starttime = System.currentTimeMillis();
       System.out.println("Randomized Hill Climbing");
       RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
       FixedIterationTrainer fit = new FixedIterationTrainer(rhc, per_iter);
       for (int i = per_iter; i <= iterations; i += per_iter) {
           System.out.println("----------" + i + " iterations-----------");
           fit.train();
           trainWriter.write(Double.toString(ef.value(rhc.getOptimal())));
           trainWriter.newLine();
           System.out.println("RHC: " + ef.value(rhc.getOptimal()));
           System.out.println("Time : " + def.format((System.currentTimeMillis() - starttime) / Math.pow(10, 3)) + " seconds");
       }
       System.out.println("============================");
       trainWriter.flush();
       trainWriter.close();

       train_file = new File("src/results/" + "flipflop_sa" + "_train.csv");
       trainWriter = new BufferedWriter(new FileWriter(train_file));

       System.out.println("Simulated Annealing");
       SimulatedAnnealing sa = new SimulatedAnnealing(100, .8, hcp);
       fit = new FixedIterationTrainer(sa, per_iter);
       starttime = System.currentTimeMillis();
       for (int i = per_iter; i <= iterations; i += per_iter) {
           System.out.println("----------" + i + " iterations-----------");
           fit.train();
           trainWriter.write(Double.toString(ef.value(sa.getOptimal())));
           trainWriter.newLine();
           System.out.println("SA: " + ef.value(sa.getOptimal()));
           System.out.println("Time : " + def.format((System.currentTimeMillis() - starttime) / Math.pow(10, 3)) + " seconds");
       }
       System.out.println("============================");
       trainWriter.flush();
       trainWriter.close();

       train_file = new File("src/results/" + "flipflop_ga" + "_train.csv");
       trainWriter = new BufferedWriter(new FileWriter(train_file));


       System.out.println("Genetic Algorithm");
       starttime = System.currentTimeMillis();
       StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200,
               100, 20, gap);
       fit = new FixedIterationTrainer(ga, per_iter);
       for(int i = per_iter; i <= iterations; i += per_iter) {
           System.out.println("----------" + i + " iterations-----------");
           fit.train();
           trainWriter.write(Double.toString(ef.value(ga.getOptimal())));
           trainWriter.newLine();
           System.out.println("GA: " + ef.value(ga.getOptimal()));
           System.out.println("Time : " + def.format((System.currentTimeMillis() - starttime) / Math.pow(10, 3)) + " seconds");
       }
       System.out.println("============================");

       trainWriter.flush();
       trainWriter.close();

       train_file = new File("src/results/" + "flipflop_mimic" + "_train.csv");
       trainWriter = new BufferedWriter(new FileWriter(train_file));

        System.out.println("MIMIC");
        MIMIC mimic = new MIMIC(100, 30, pop);
        fit = new FixedIterationTrainer(mimic, 1);
        starttime = System.currentTimeMillis();
        for(int i = 1; i <= iterations / per_iter; i += 1)
        {
            System.out.println("----------" + i + " iterations-----------");
            fit.train();
            trainWriter.write(Double.toString(ef.value(mimic.getOptimal())));
            trainWriter.newLine();
            System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));
            System.out.println("Time : " + def.format((System.currentTimeMillis() - starttime) / Math.pow(10, 3)) + " seconds");
        }
        trainWriter.flush();
        trainWriter.close();
    }
}
