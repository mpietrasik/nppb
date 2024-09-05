import java.io.*;
import cc.mallet.types.*;
import cc.mallet.util.Randoms;

public class Main {
    public static void main(String[] args) throws IOException {

        InstanceList instances = InstanceList.load(new File("carnivora.mallet"));
        InstanceList testing = InstanceList.load(new File("carnivora.mallet"));

        Randoms random = new Randoms();

        NPPB sa = new NPPB(instances, testing, 3, random, 10.0, 0.01, 0.01);
        sa.learn(5000);
    }
}