package project2;

import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Random;

import opt.ga.MaxKColorFitnessFunction;
import opt.ga.Vertex;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.SwapNeighbor;
import opt.example.FourPeaksEvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import java.io.*;
/**
 * 
 * @author kmandal
 * @version 1.0
 */
public class MaxKColoringTest {
    /** The n value */
    private static final int N = 500; // number of vertices
    private static final int L = 100; // L adjacent nodes per vertex
    private static final int K = 100; // K possible colors
    private static DecimalFormat def = new DecimalFormat("0.000");
    /**
     * The test main
     */
    public static void mkTest(int input, int iterations) throws IOException {
//        int N = input;
//        int N = 10000;
        Random random = new Random(System.currentTimeMillis());
        // create the random velocity
        Vertex[] vertices = new Vertex[N];
        for (int i = 0; i < N; i++) {
            Vertex vertex = new Vertex();
            vertices[i] = vertex;
            vertex.setAdjMatrixSize(L);
            for (int j = 0; j < L; j++) {
                vertex.getAadjacencyColorMatrix().add(random.nextInt(N * L));
            }
        }
        /*for (int i = 0; i < N; i++) {
            Vertex vertex = vertices[i];
            System.out.println(Arrays.toString(vertex.getAadjacencyColorMatrix().toArray()));
        }*/
        // for rhc, sa, and ga we use a permutation based encoding
        MaxKColorFitnessFunction ef = new MaxKColorFitnessFunction(vertices);
        Distribution odd = new DiscretePermutationDistribution(K);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new SingleCrossOver();
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

        Distribution df = new DiscreteDependencyTree(.1);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        File train_file = new File("src/results/" + "maxk_rhc" + "_train.csv");
        BufferedWriter trainWriter;
        trainWriter = new BufferedWriter(new FileWriter(train_file));
        int per_iter = 5;

        // Randomized Hill Climbing
        long starttime = System.currentTimeMillis();
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, per_iter);
        System.out.println("Randomized Hill Climbing");
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

        train_file = new File("src/results/" + "maxk_sa" + "_train.csv");
        trainWriter = new BufferedWriter(new FileWriter(train_file));
        // Simulated Annealing
        starttime = System.currentTimeMillis();
        SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .8, hcp);
        fit = new FixedIterationTrainer(sa, per_iter);
        System.out.println("Simulated Annealing");
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

        train_file = new File("src/results/" + "maxk_ga" + "_train.csv");
        trainWriter = new BufferedWriter(new FileWriter(train_file));

        // Genetic Algorithm
        starttime = System.currentTimeMillis();
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(100, 10, 60, gap);
        fit = new FixedIterationTrainer(ga, per_iter);
        System.out.println("Genetic Algorithm");
        for (int i = per_iter; i <= iterations; i += per_iter) {
            System.out.println("----------" + i + " iterations-----------");
//            fit.train();
            trainWriter.write(Double.toString(ef.value(ga.getOptimal())));
            trainWriter.newLine();
            System.out.println("GA: " + ef.value(ga.getOptimal()));
            System.out.println("Time : " + def.format((System.currentTimeMillis() - starttime) / Math.pow(10, 3)) + " seconds");
        }
        System.out.println("============================");
        trainWriter.flush();
        trainWriter.close();

        train_file = new File("src/results/" + "maxk_mimic" + "_train.csv");
        trainWriter = new BufferedWriter(new FileWriter(train_file));

        // MIMIC
        starttime = System.currentTimeMillis();
        MIMIC mimic = new MIMIC( 50, 15, pop);
        fit = new FixedIterationTrainer(mimic, 1);
        System.out.println("MIMIC");
        for (int i = 1; i <= iterations/per_iter; i += 1) {
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
    
    public static void main(String[] args) throws IOException {
    	mkTest(200, 200);
    	System.out.println("Done!");
    }
}
