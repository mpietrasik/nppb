import com.rits.cloning.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.io.*;
import java.util.HashMap;
import java.lang.Math;
import cc.mallet.types.*;
import cc.mallet.util.Randoms;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.map.hash.TObjectDoubleHashMap;

public class NPPB{

    InstanceList instances, testing;
    Randoms random;

    Tree current_tree;

    double alpha;
    double gamma;
    double eta;
    double temperature;

    final double MAX_TEMP = 2.5;

    String relationsFile;
    String treeFile;

    // Initialize NPPB class
    public NPPB (InstanceList instances, InstanceList testing, int numLevels, Randoms random, double alpha, double gamma, double eta) {
        this.alpha = alpha;
        this.gamma = gamma;
        this.eta = eta;

        this.random = random;
        this.temperature = MAX_TEMP;
        this.instances = instances;

        int simulation_identifier = (new Randoms ()).nextInt(10000000);
        this.relationsFile = "axioms_" + Integer.toString(simulation_identifier) + ".out";
        this.treeFile = "tree_"+ Integer.toString(simulation_identifier) + ".out";

        this.current_tree = new Tree();
        this.current_tree.initialize(instances, testing, numLevels, random);

    }

    // Perform simulated annealing
    public void learn(int numIterations) throws IOException, FileNotFoundException{

        Cloner cloner = new Cloner();

        final long startTime = System.currentTimeMillis();
        System.out.println("Training starting...");
        for (int iteration = 1; iteration <= numIterations; iteration++) {

            this.temperature = updateTemperature(this.temperature, numIterations, iteration);
            Tree clone = cloner.deepClone(this.current_tree);
            int changePath;
            if (Math.random() > 0.9){
                changePath = this.current_tree.choosePathToUpdate();
            }else{
                changePath = random.nextInt(this.instances.size());
            }

            int changeWord = random.nextInt(this.instances.getDataAlphabet().size());

            if (iteration % 2 == 0) {
                clone.sampleTopics(changeWord);
            }else{
                clone.samplePath(changePath, iteration);
            }

            if (acceptance(this.current_tree.calculateObjectiveFunction(), clone.calculateObjectiveFunction(), temperature)) {
                this.current_tree = clone;
            }
            if (iteration % 1000 == 0) {
                System.out.println("Iteration: " + iteration + "\t Objective function: " + this.current_tree.calculateObjectiveFunction());
            }
        }

        System.out.println("Training complete");
        final long endTime = System.currentTimeMillis();
        System.out.println("Total execution time (minutes): " + ((double) (endTime - startTime) / 60000.0));

        this.current_tree.writeRelationsToFile(this.relationsFile);
        this.current_tree.writeTreeToFile(this.treeFile);
    }

    public boolean acceptance(double oldObjective, double newObjective, double temperature){
        if (newObjective > oldObjective){
            return true;
        }
        double acceptanceProbability = Math.exp(-1 * ((oldObjective - newObjective) / temperature));
        return Math.random() < acceptanceProbability;
    }

    public double updateTemperature(double old_temperature, int numIts, int currentIteration){
        double new_temperature = MAX_TEMP * Math.pow(0.01 / (double) MAX_TEMP, (double) currentIteration / (double) numIts);
        return new_temperature;
    }
}

class Tree {

    InstanceList instances;
    InstanceList testing;

    NCRPNode rootNode, node;

    int numLevels;
    int numDocuments;
    int numTypes;

    double alpha;
    double gamma;
    double eta;
    double etaSum;

    int[][] levels; // indexed < doc, token >
    NCRPNode[] documentLeaves;
    Alphabet alphabet;

    int totalNodes = 0;

    Randoms random;

    int numWordsToDisplay = 10;

    HashMap<Integer, ArrayList<Integer>> docs_with_word;
    HashMap<Integer, ArrayList<NCRPNode>> word_paths;
    HashMap<Integer, NCRPNode> word_nodes;
    int[] word_levels;

    public Tree () {
        alpha = 30.0;
        gamma = 0.1;
        eta = 0.1;
    }

    public void initialize(InstanceList instances, InstanceList testing, int numLevels, Randoms random) {
        this.instances = instances;
        this.testing = testing;
        this.numLevels = numLevels;
        this.random = random;
        word_paths = new HashMap<Integer, ArrayList<NCRPNode>>();
        word_nodes= new HashMap<Integer, NCRPNode>();
        alphabet = instances.getDataAlphabet();
        word_levels = new int[alphabet.size()];

        if (! (instances.get(0).getData() instanceof FeatureSequence)) {
            throw new IllegalArgumentException("Input must be a FeatureSequence, using the --feature-sequence option when impoting data, for example");
        }

        numDocuments = instances.size();
        numTypes = instances.getDataAlphabet().size();

        etaSum = eta * numTypes;

        NCRPNode[] path = new NCRPNode[numLevels];

        rootNode = new NCRPNode(alphabet.size(), numTypes);

        levels = new int[numDocuments][];
        documentLeaves = new NCRPNode[numDocuments];

        int word;
        for (int doc=0; doc < numDocuments; doc++) {
            FeatureSequence fs = (FeatureSequence) instances.get(doc).getData();

            int seqLen = fs.getLength();

            path[0] = rootNode;
            rootNode.customers++;
            for (int word_index = 0; word_index <fs.getLength(); word_index++){
                word = fs.getIndexAtPosition(word_index);
                rootNode.customers_with_word[word]++;
            }
            for (int level = 1; level < numLevels; level++) {
                path[level] = path[level-1].select();
                path[level].customers++;
                for (int word_index = 0; word_index <fs.getLength(); word_index++) {
                    word = fs.getIndexAtPosition(word_index);
                    path[level].customers_with_word[word]++;
                }
            }

            node = path[numLevels - 1];

            for (int word_index = 0; word_index < fs.getLength(); word_index++) {
                int type = fs.getIndexAtPosition(word_index);
                if (word_paths.containsKey(type)) {
                    word_paths.get(type).add(node);
                } else {
                    word_paths.put(type, new ArrayList<NCRPNode>());
                    word_paths.get(type).add(node);
                }
            }
            levels[doc] = new int[seqLen];
            documentLeaves[doc] = node;
        }

        // Initialize levels
        for (int type = 0; type < alphabet.size() ; type++){

            int number_of_paths = word_paths.get(type).size();
            int sampled_path_for_word = random.nextInt(number_of_paths);
            node = word_paths.get(type).get(sampled_path_for_word);

            NCRPNode[] path_for_word = new NCRPNode[numLevels];
            int level;
            for (level = numLevels - 1; level >= 0; level--) {
                path_for_word[level] = node;
                node = node.parent;
            }

            word_levels[type] = random.nextInt(numLevels);
            node = path_for_word[word_levels[type]];
            node.totalTokens++;
            node.typeCounts[type]++;
            word_nodes.put(type, node);
        }

        docs_with_word = new HashMap<Integer, ArrayList<Integer>>();

        for (int doc=0; doc < numDocuments; doc++) {
            FeatureSequence fs = (FeatureSequence) instances.get(doc).getData();

            for (int word_index = 0; word_index < fs.getLength(); word_index++) {
                word = fs.getIndexAtPosition(word_index);
                if (docs_with_word.containsKey(doc)) {
                    docs_with_word.get(doc).add(word);
                }
                else{
                    docs_with_word.put(doc, new ArrayList<Integer>());
                    docs_with_word.get(doc).add(word);
                }
            }
        }
    }

    public void samplePath(int doc, int iteration) {
        NCRPNode[] path = new NCRPNode[numLevels];
        NCRPNode node;
        int level, token, type, topicCount;
        double weight;

        FeatureSequence fs = (FeatureSequence) instances.get(doc).getData();

        node = documentLeaves[doc];
        for (level = numLevels - 1; level >= 0; level--) {
            path[level] = node;
            node = node.parent;
        }

        documentLeaves[doc].dropPath(doc);

        for (int word_index = 0; word_index < fs.getLength(); word_index++) {
            type = fs.getIndexAtPosition(word_index);
            word_paths.get(type).remove(documentLeaves[doc]);
        }

        TObjectDoubleHashMap<NCRPNode> nodeWeights =
                new TObjectDoubleHashMap<NCRPNode>();


        double objectiveValue = 0;
        level = 0;
        calculatePathObjective(nodeWeights, rootNode, objectiveValue, doc, level);
        NCRPNode[] nodes = nodeWeights.keys(new NCRPNode[] {});

        TIntIntHashMap[] typeCounts = new TIntIntHashMap[numLevels];


        for (level = 0; level < numLevels; level++) {
            typeCounts[level] = new TIntIntHashMap();
        }

        nodes = nodeWeights.keys(new NCRPNode[] {});
        double[] weights = new double[nodes.length];
        double sum = 0.0;
        double max = Double.NEGATIVE_INFINITY;

        // To avoid underflow, we're using log weights and normalizing the node weights so that the largest weight is always 1.
        for (int i=0; i<nodes.length; i++) {
            if (nodeWeights.get(nodes[i]) > max) {
                max = nodeWeights.get(nodes[i]);
            }
        }

        for (int i=0; i<nodes.length; i++) {
            weights[i] =  1;
            sum += weights[i];
        }

        node = nodes[ random.nextDiscrete(weights, sum) ];
        if (! node.isLeaf()) {
            node = node.getNewLeaf();
        }

        node.addPath(doc);
        documentLeaves[doc] = node;

        for (int word_index = 0; word_index < fs.getLength(); word_index++) {
            type = fs.getIndexAtPosition(word_index);
            word_paths.get(type).add(node);
        }
    }

    public void calculatePathObjective(TObjectDoubleHashMap<NCRPNode> nodeWeights,  NCRPNode node, double objectiveValue, int doc, int level){

        for (NCRPNode child: node.children) {
            calculatePathObjective(nodeWeights, child, objectiveValue, doc, level + 1);
        }

        if (level == (numLevels -1)){
            objectiveValue = 1.0;
        }
        else{
            objectiveValue = 0.1;
        }

        nodeWeights.put(node, objectiveValue);
    }

    public void calculateNCRP(TObjectDoubleHashMap<NCRPNode> nodeWeights, NCRPNode node, double weight) {
        for (NCRPNode child: node.children) {
            calculateNCRP(nodeWeights, child,
                    weight + Math.log((double) child.customers / (node.customers + gamma)));
        }

        nodeWeights.put(node, weight + Math.log(gamma / (node.customers + gamma)));
    }

    public void calculateWordLikelihood(TObjectDoubleHashMap<NCRPNode> nodeWeights, NCRPNode node, double weight, TIntIntHashMap[] typeCounts, double[] newTopicWeights, int level, int iteration) {

        double nodeWeight = 0.0;
        int[] types = typeCounts[level].keys();
        int totalTokens = 0;

        for (int type: types) {
            for (int i=0; i<typeCounts[level].get(type); i++) {
                nodeWeight +=
                        Math.log((eta + node.typeCounts[type] + i) /
                                (etaSum + node.totalTokens + totalTokens));
                totalTokens++;
            }
        }

        for (NCRPNode child: node.children) {
            calculateWordLikelihood(nodeWeights, child, weight + nodeWeight, typeCounts, newTopicWeights, level + 1, iteration);
        }

        level++;
        while (level < numLevels) {
            nodeWeight += newTopicWeights[level];
            level++;
        }
        nodeWeights.adjustValue(node, nodeWeight);

    }

    public void sampleTopics(int word) {

        NCRPNode[] word_path = new NCRPNode[numLevels];
        NCRPNode node;
        int[] levelCounts = new int[numLevels];
        int level;
        double sum;

        int number_of_paths = word_paths.get(word).size();
        int sampled_path_for_word = random.nextInt(number_of_paths);
        node = word_paths.get(word).get(sampled_path_for_word);

        NCRPNode[] path_for_word = new NCRPNode[numLevels];
        for (level = numLevels - 1; level >= 0; level--) {
            path_for_word[level] = node;
            node = node.parent;
        }

        NCRPNode current_node_for_word = word_nodes.get(word);
        current_node_for_word.totalTokens--;
        current_node_for_word.typeCounts[word]--;
        assert(current_node_for_word.totalTokens > 0);
        assert(current_node_for_word.typeCounts[word] > 0);

        int total_path_word_count = 0;
        for (level=0; level < numLevels; level++) { total_path_word_count += path_for_word[level].totalTokens; }

        double[] level_weights = new double[numLevels];
        sum = 0.0;
        double log_prior_for_level = 0;
        double log_likelihood_for_level;
        int words_seen_so_far = 0;
        double eta_small = 0.01;

        // OLD WAY OF CALCULATING PRIOR
        for (level=0; level < numLevels; level++) {
            log_prior_for_level = 0;
            words_seen_so_far += path_for_word[level].totalTokens;

            log_likelihood_for_level =  Math.log((eta + path_for_word[level].customers_with_word[word]) / (etaSum + path_for_word[level].customers));
             level_weights[level] = Math.exp(log_prior_for_level + log_likelihood_for_level);
        }
        //

        double max = Double.NEGATIVE_INFINITY;
        for (int i=0; i<level_weights.length; i++) {
            if (level_weights[i] > max) {
                max = level_weights[i];
            }
        }

        for (int i=0; i<level_weights.length; i++) {
            level_weights[i] = Math.exp(level_weights[i] - max);
            sum += level_weights[i];
        }

        node = path_for_word[ random.nextDiscrete(level_weights, sum) ];

        node.totalTokens++;
        node.typeCounts[word]++;
        word_nodes.put(word, node);
        word_levels[word] = node.level;

        if ((current_node_for_word.totalTokens == 0) && (current_node_for_word.customers == 0)){
            current_node_for_word.dropPath_word(word);
        }
    }

    public void writeRelationsToFile(String relationsFile) throws IOException, FileNotFoundException {
        StringBuffer output_string = writeRelationsToFile(new StringBuffer(), this.rootNode);
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(relationsFile)));
        System.out.println(relationsFile);
        pw.print(output_string);
        pw.close();
    }

    public StringBuffer writeRelationsToFile(StringBuffer output_string, NCRPNode node) throws IOException {

        for (int i = 0; i < node.typeCounts.length; i++){
            for (int j = 0; j < node.typeCounts.length; j++) {
                if (i < j) {
                    assert ((node.typeCounts[i] == 0 || node.typeCounts[i] == 1) && (node.typeCounts[j] == 0 || node.typeCounts[j] == 1));
                    if ((node.typeCounts[i] == 1) && (node.typeCounts[j] == 1)) {
                        output_string.append(alphabet.lookupObject(i) + " " + alphabet.lookupObject(j) + "\n" );
                    }
                }
            }
        }

        if (node.level < 10000) {
            for (NCRPNode child : node.children) {
                for (int i = 0; i < node.typeCounts.length; i++) {
                    for (int j = 0; j < child.typeCounts.length; j++) {
                        assert ((node.typeCounts[i] == 0 || node.typeCounts[i] == 1) && (child.typeCounts[j] == 0 || child.typeCounts[j] == 1));
                        if ((node.typeCounts[i] == 1) && (child.typeCounts[j] == 1)) {
                            output_string.append(alphabet.lookupObject(i) + " " + alphabet.lookupObject(j) + "\n");
                        }
                    }
                }
            }

            for (NCRPNode child : node.children) {
                output_string = writeRelationsToFile(output_string, child);
            }
        }
        return output_string;
    }

    public void writeTreeToFile(String tree_file) throws IOException, FileNotFoundException {
        StringBuffer output_string = writeTreeToFile(new StringBuffer(), this.rootNode, 0);
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(tree_file)));
        pw.print(output_string);
        pw.close();
    }

    public StringBuffer writeTreeToFile(StringBuffer output_string, NCRPNode node, int indent ) {
        for (int i = 0; i < indent; i++) {
            output_string.append("    ");
        }

        output_string.append("[ ");
        output_string.append(Integer.toString(node.nodeID));
        output_string.append(" ] ");
        output_string.append(Integer.toString(node.customers));
        output_string.append(" , ");
        output_string.append(Integer.toString(node.totalTokens));
        output_string.append(" : ");
        for (int i = 0; i < node.typeCounts.length; i++){
            assert(node.typeCounts[i] == 0 || node.typeCounts[i] == 1);
            if (node.typeCounts[i] == 1){
                output_string.append(alphabet.lookupObject(i));
                output_string.append(" ");
            }
        }
        output_string.append("\n");
        for (NCRPNode child : node.children) {
            writeTreeToFile(output_string, child, indent + 1);
        }

        return output_string;
    }

    public void printTree(NCRPNode node, int indent ) {
        StringBuffer out = new StringBuffer();
        for (int i = 0; i < indent; i++) {
            out.append("    ");
        }

        out.append(" [ ");
        out.append(Integer.toString(node.nodeID));
        out.append(" ] ");
        out.append(Integer.toString(node.customers));
        out.append(" , ");
        out.append(Integer.toString(node.totalTokens));
        out.append(" : ");
        for (int i = 0; i < node.typeCounts.length; i++){
            assert(node.typeCounts[i] == 0 || node.typeCounts[i] == 1);
            if (node.typeCounts[i] == 1){
                out.append(alphabet.lookupObject(i));
                out.append(" ");
            }
        }
        out.append(Arrays.toString(node.customers_with_word));
        System.out.println(out);
        for (NCRPNode child : node.children) {
            printTree(child, indent + 1);
        }
    }

    public void printNode(NCRPNode node, int indent, boolean withWeight) {
        StringBuffer out = new StringBuffer();
        for (int i=0; i<indent; i++) {
            out.append("    ");
        }

        out.append(node.nodeID + " " + node.totalTokens + "/" + node.customers + " ");
        out.append(node.getTopWords(numWordsToDisplay, withWeight));
        System.out.println(out);

        for (NCRPNode child: node.children) {
            printNode(child, indent + 1, withWeight);
        }
    }

    public StringBuffer printTopic(NCRPNode node, StringBuffer str){
        str.append(node.nodeID + "|");
        str.append(node.getTopWords(numWordsToDisplay, true) + "\n");
        for (NCRPNode child: node.children) {
            str = printTopic(child, str);
        }
        return str;
    }

    public StringBuffer printPath(NCRPNode node, ArrayList<Integer> path, StringBuffer str){
        path.add(node.nodeID );
        for (NCRPNode child: node.children) {
            str = printPath(child, path, str);
        }
        if (node.children.size() == 0){
            str.append(path);
            str.append("\n");
        }
        path.remove(path.size() - 1);

        return str;
    }

    public double calculateObjectiveFunction(){

        int word, level;
        double objective = 0;
        for (int doc = 0; doc < numDocuments; doc++) {
            FeatureSequence fs = (FeatureSequence) instances.get(doc).getData();
            node = documentLeaves[doc];
            for (level = numLevels - 1; level >= 0; level--) {
                boolean hasWord = false;
                for (int word_index = 0; word_index < fs.getLength(); word_index++) {
                    word = fs.getIndexAtPosition(word_index);
                    if (node.typeCounts[word] == 1) {
                        hasWord = true;
                        objective += 1.0 / node.totalTokens;
                    }
                }
                if (!hasWord) {
                    objective -= (double) node.totalTokens + 2;
                }

                if (node.totalTokens > 1) {
                    objective -= 3 + node.totalTokens * (numLevels - level - 1) * 1;
                }
                node = node.parent;
            }
        }

        return objective;
    }

    public int choosePathToUpdate() {
        int level, word;
        double sum = 0;
        double[] doc_probabilities = new double[numDocuments];
        double current_max = Double.NEGATIVE_INFINITY;
        int arg_max = 0;
        for (int doc = 0; doc < numDocuments; doc++) {
            int number_of_words_in_doc_path = 0;
            FeatureSequence fs = (FeatureSequence) instances.get(doc).getData();
            node = documentLeaves[doc];
            for (level = numLevels - 1; level >= 0; level--) {
                for (int word_index = 0; word_index < fs.getLength(); word_index++) {
                    word = fs.getIndexAtPosition(word_index);
                    if (node.typeCounts[word] == 1) {
                        number_of_words_in_doc_path++;
                    }
                }
            }
            doc_probabilities[doc] = -1 * Math.log(0.00001 + (double) number_of_words_in_doc_path / (double) fs.size());
            sum += doc_probabilities[doc];
            if (doc_probabilities[doc] > current_max){
                arg_max = doc;
                current_max = doc_probabilities[doc];
            }
        }
        return arg_max;
    }

    class NCRPNode {
        int customers;
        ArrayList<NCRPNode> children;
        NCRPNode parent;
        int level;

        int totalTokens;
        int[] typeCounts;
        int vocabulary_size;

        public int[] customers_with_word;

        public int nodeID;

        public NCRPNode(NCRPNode parent, int vocabulary_size, int dimensions, int level) {
            customers = 0;
            this.parent = parent;
            this.vocabulary_size = vocabulary_size;
            this.customers_with_word = new int[vocabulary_size];

            children = new ArrayList<NCRPNode>();
            this.level = level;

            totalTokens = 0;
            typeCounts = new int[dimensions];

            nodeID = totalNodes;
            totalNodes++;
        }

        public NCRPNode(int vocabulary_size, int dimensions) {
            this(null, vocabulary_size, dimensions, 0);
        }

        public NCRPNode addChild() {
            NCRPNode node = new NCRPNode(this, vocabulary_size, typeCounts.length, level + 1);
            children.add(node);
            return node;
        }

        public boolean isLeaf() {
            return level == numLevels - 1;
        }

        public NCRPNode getNewLeaf() {
            NCRPNode node = this;
            for (int l=level; l<numLevels - 1; l++) {
                node = node.addChild();
            }
            return node;
        }

        public void dropPath(int doc) {
            int word;
            NCRPNode node = this;
            node.customers--;
            FeatureSequence fs = (FeatureSequence) instances.get(doc).getData();
            for (int word_index = 0; word_index <fs.getLength(); word_index++) {
                word = fs.getIndexAtPosition(word_index);
                node.customers_with_word[word]--;
                assert(node.customers_with_word[word] >= 0);
            }

            boolean remove_up_tree = true;
            if (node.customers == 0 && node.totalTokens == 0) {
                node.parent.remove(node);
            }
            else{ remove_up_tree = false; }
            for (int l = 1; l < numLevels; l++) {
                node = node.parent;
                node.customers--;
                for (int word_index = 0; word_index <fs.getLength(); word_index++) {
                    word = fs.getIndexAtPosition(word_index);
                    node.customers_with_word[word]--;
                    assert(node.customers_with_word[word] >= 0);
                }

                if (node.customers == 0 && node.totalTokens == 0 && remove_up_tree) {
                    node.parent.remove(node);
                }
                else{ remove_up_tree = false; }
            }
        }

        public void dropPath_word(int word){
            NCRPNode node = this;
            boolean drop_childen = true;
            if (drop_childen) {
                node.children.clear();
                if ((node.totalTokens == 0) && (node.customers == 0)){
                    node.parent.remove(node);
                }
                for (int l = node.level; l > 0; l--){
                    node = node.parent;
                    if ((node.totalTokens == 0) && (node.customers == 0)){
                        node.parent.remove(node);
                    }
                }
            }
        }

        //this might not be necessary becasue it would all be removed in path sampling... if there are still child nodes it means you cannot remove
        public boolean drop_children_crawl(NCRPNode node){
            if (node.totalTokens > 0){
                return false;
            }
            for (NCRPNode child: children) {
                if (!drop_children_crawl(child)){
                    return false;
                }
            }
            return true;
        }

        public void remove(NCRPNode node) {
            children.remove(node);
        }

        public void addPath(int doc) {
            int word;
            NCRPNode node = this;
            node.customers++;
            FeatureSequence fs = (FeatureSequence) instances.get(doc).getData();
            for (int word_index = 0; word_index <fs.getLength(); word_index++) {
                word = fs.getIndexAtPosition(word_index);
                node.customers_with_word[word]++;
            }
            for (int l = 1; l < numLevels; l++) {
                node = node.parent;
                node.customers++;
                for (int word_index = 0; word_index <fs.getLength(); word_index++) {
                    word = fs.getIndexAtPosition(word_index);
                    node.customers_with_word[word]++;
                }
            }
        }

        public NCRPNode select() {
            double[] weights = new double[children.size() + 1];

            weights[0] = gamma / (gamma + customers);

            int i = 1;
            for (NCRPNode child: children) {
                weights[i] = (double) child.customers / (gamma + customers);
                i++;
            }

            int choice = random.nextDiscrete(weights);
            if (choice == 0) {
                return(addChild());
            }
            else {
                return children.get(choice - 1);
            }
        }

        public String getTopWords(int numWords, boolean withWeight) {
            IDSorter[] sortedTypes = new IDSorter[numTypes];

            for (int type=0; type < numTypes; type++) {
                sortedTypes[type] = new IDSorter(type, typeCounts[type]);
            }
            Arrays.sort(sortedTypes);
            withWeight = true;
            //Alphabet alphabet = instances.getDataAlphabet();
            StringBuffer out = new StringBuffer();
            for (int i = 0; i < numWords; i++) {
                if (withWeight){
                    out.append(alphabet.lookupObject(sortedTypes[i].getID()) + ":" + sortedTypes[i].getWeight() + " ");
                }else
                    out.append(alphabet.lookupObject(sortedTypes[i].getID()) + " ");
            }
            return out.toString();
        }

        @Override
        public String toString()
        {
            return Integer.toString(this.nodeID);
        }

    }
}
