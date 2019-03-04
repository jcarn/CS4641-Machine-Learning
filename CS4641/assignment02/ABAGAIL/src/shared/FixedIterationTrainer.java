package shared;

import java.util.*;

/**
 * A fixed iteration trainer
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class FixedIterationTrainer implements Trainer {
    
    /**
     * The inner trainer
     */
    private Trainer trainer;
    
    /**
     * The number of iterations to train
     */
    private int iterations;
    
    /**
     * Make a new fixed iterations trainer
     * @param t the trainer
     * @param iter the number of iterations
     */
    public FixedIterationTrainer(Trainer t, int iter) {
        trainer = t;
        iterations = iter;
    }

    /**
     * @see shared.Trainer#train()
     */
    public double train() {
        double sum = 0;
        for (int i = 0; i < iterations; i++) {
            sum += trainer.train();
//            if (i % 10 == 0) {
//                System.out.println(sum);
//            }
//            if (i % 200 == 0) {
//                Scanner s = new Scanner(System.in);
//                s.next();
//            }

        }
        return sum / iterations;
    }
    

}
