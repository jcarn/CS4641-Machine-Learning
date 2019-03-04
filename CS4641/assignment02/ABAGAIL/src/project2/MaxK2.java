package project2;

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

/**
 *
 * @author kmandal
 * @version 1.0
 */
public class MaxK2{
    /** The n value */
    private static int N = 500; // number of vertices
    private static int L = 40; // L adjacent nodes per vertex
    private static int K = 40; // K possible colors
    private static int I = 10;// number of iterations for averaging
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
//        if(args.length < 4)
//        {
//            System.out.println("Provide number of vertices, adjacent" +
//                    "nodes per vertex, number of possible colors, and number of iterations.");
//            System.exit(0);
//        }
//        N = Integer.parseInt(args[0]);
//        L = Integer.parseInt(args[1]);
//        K = Integer.parseInt(args[2]);
//        I = Integer.parseInt(args[3]);

        if(N < 0 || L < 0 || K < 0 || I < 0)
        {
            System.out.println(" Cannot be negaitve.");
            System.exit(0);
        }

        int n = N;
        int l = L;
        int k = K;
//        for (int n = 8; n <= N; n*=2){
//            for (int l = 4; l <= n; l*=2){
//                for (int k = 2; k <= l; k*=2){

                    Random random = new Random(n*l);
                    // create the random velocity
                    Vertex[] vertices = new Vertex[n];
                    for (int i = 0; i < n; i++) {
                        Vertex vertex = new Vertex();
                        vertices[i] = vertex;
                        vertex.setAdjMatrixSize(l);
                        for(int j = 0; j <l; j++ ){
                            vertex.getAadjacencyColorMatrix().add(random.nextInt(n*l));
                        }
                    }
                    /*for (int i = 0; i < N; i++) {
                        Vertex vertex = vertices[i];
                        System.out.println(Arrays.toString(vertex.getAadjacencyColorMatrix().toArray()));
                    }*/
                    // for rhc, sa, and ga we use a permutation based encoding
                    MaxKColorFitnessFunction ef = new MaxKColorFitnessFunction(vertices);
                    Distribution odd = new DiscretePermutationDistribution(k);
                    NeighborFunction nf = new SwapNeighbor();
                    MutationFunction mf = new SwapMutation();
                    CrossoverFunction cf = new SingleCrossOver();
                    HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
                    GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

                    Distribution df = new DiscreteDependencyTree(.1);
                    ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

                    long starttime = System.currentTimeMillis();
                    System.out.printf("Current Input Sizes: %d %d %d\n", n, l, k);
                    System.out.println("Randomized Hill Climbing\n============================");
                    double sumEf   = 0;
                    double sumTime = 0;
                    for(int i = 0; i < I; i++)
                    {
                        long ti = System.nanoTime();
                        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
                        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 20000);
                        fit.train();
                        System.out.println(ef.value(rhc.getOptimal()) + ", " + (((double)(System.nanoTime() - ti))/ 1e9d));
//                        System.out.println("RHC: " + ef.value(rhc.getOptimal()));
//                        System.out.println(ef.foundConflict());
                        //System.out.println("Time : "+ (System.currentTimeMillis() - starttime));
                        sumEf   += ef.value(rhc.getOptimal());
                        sumTime += ((double)(System.nanoTime() - ti))/1e9d;
                    }
                    double efAverage   = sumEf   / I;
                    double timeAverage = sumTime / I;
                    System.out.printf("Ef Average: %f %n", efAverage);
                    System.out.printf("Time Average: %f %n", timeAverage);

                    System.out.println("Simulated Annealing \n============================");
                    sumEf   = 0;
                    sumTime = 0;
                    for(int i = 0; i < I; i++)
                    {
                        long ti = System.nanoTime();
                        SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .1, hcp);
                        FixedIterationTrainer fit = new FixedIterationTrainer(sa, 20000);
                        fit.train();
                        System.out.println(ef.value(sa.getOptimal()) + ", " + (((double)(System.nanoTime() - ti))/ 1e9d));
                        // System.out.println("SA: " + ef.value(sa.getOptimal()));
                        // System.out.println(ef.foundConflict());
                        // System.out.println("Time : "+ (System.currentTimeMillis() - starttime));
                        sumEf   += ef.value(sa.getOptimal());
                        sumTime += ((double)(System.nanoTime() - ti))/1e9d;
                    }
                    efAverage   = sumEf   / I;
                    timeAverage = sumTime / I;
                    System.out.printf("Ef Average: %f %n", efAverage);
                    System.out.printf("Time Average: %f %n", timeAverage);

                    System.out.println("Genetic Algorithm\n============================");
                    sumEf   = 0;
                    sumTime = 0;
                    for(int i = 0; i < I; i++)
                    {
                        long ti = System.nanoTime();
                        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 10, 60, gap);
                        FixedIterationTrainer fit = new FixedIterationTrainer(ga, 50);
                        fit.train();
                        System.out.println(ef.value(ga.getOptimal()) + ", " + (((double)(System.nanoTime() - ti))/ 1e9d));
                        // System.out.println("GA: " + ef.value(ga.getOptimal()));
                        // System.out.println(ef.foundConflict());
                        // System.out.println("Time : "+ (System.currentTimeMillis() - starttime));
                        sumEf   += ef.value(ga.getOptimal());
                        sumTime += ((double)(System.nanoTime() - ti))/1e9d;
                    }
                    efAverage   = sumEf   / I;
                    timeAverage = sumTime / I;
                    System.out.printf("Ef Average: %f %n", efAverage);
                    System.out.printf("Time Average: %f %n", timeAverage);

                    System.out.println("MIMIC\n============================");
                    sumEf   = 0;
                    sumTime = 0;
                    for(int i = 0; i < I; i++)
                    {
                        long ti = System.nanoTime();
                        MIMIC mimic = new MIMIC(200, 100, pop);
                        FixedIterationTrainer fit = new FixedIterationTrainer(mimic, 5);
                        fit.train();
                        System.out.println(ef.value(mimic.getOptimal()) + ", " + (((double)(System.nanoTime() - ti))/ 1e9d));
                        // System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));
                        // System.out.println(ef.foundConflict());
                        // System.out.println("Time : "+ (System.currentTimeMillis() - starttime));
                        sumEf   += ef.value(mimic.getOptimal());
                        sumTime += ((double)(System.nanoTime() - ti))/1e9d;
                    }
                    efAverage   = sumEf   / I;
                    timeAverage = sumTime / I;
                    System.out.printf("Ef Average: %f %n", efAverage);
                    System.out.printf("Time Average: %f %n", timeAverage);
                }
//            }
//        }
//    }
}