import java.util.ArrayList;
import java.util.Arrays;
import java.util.*;


// Building MST using Kruskal algorithm and prim algorithm
// The work is inspired by the tutorial materials and lectures from Prof Weifa Liang, ANU
// By Isaac Chen, 2018

class MST {
    // Using Kruskal algorithm to build MST
    public static class KruskalMST {
    public List<Edge> mst = new ArrayList<Edge>();
    public double weight() {
        double weight = 0;
        for (Edge e: mst) weight+=e.weight;
        return weight;
    }
    // Given a complete undirected graph, generate a MST
    // The graph is in symmetric matrix representation
    // Constructor for symmetric matrix graph
    public KruskalMST(Edge [][] graph) {
        int edge_counter = 0;
        int size = graph.length;
        DisjointUnionSets ds = new DisjointUnionSets(size);
        Edge[] edges = new Edge [size * (size - 1) /2];
        for (int i = 0; i < size; i++)
            for (int j = i; j < size; j++)
                if (i != j) {edges[edge_counter] = graph[i][j]; edge_counter++;}
                // the graph contains the edges
                // only need to generate edge for one side as the matrix is symmetric
        Arrays.sort(edges); // the sorting for arrays in java by default is quick sorting
        for (int i = 0; i<edge_counter;i++) {
            if (mst.size()>=size - 1) break; // number of edges in mst must not exceed number of nodes
            Node v = edges[i].v;
            int parent1 = ds.find(v.id);
            Node u = edges[i].u;
            int parent2 = ds.find(u.id);
                // use Disjointset to build the MST
            if (parent1!=parent2) {
                ds.union(v.id, u.id);
                mst.add(edges[i]);
            }
        }
    }
        // Given a random connected undirected graph, generate a MST
        // The graph is in linked list representation
        // Constructor for linked list graph
        public KruskalMST(LinkedList<Node>[] graph) {
            int edge_counter = 0;
            int size = graph.length;
            int esize = 0;
            DisjointUnionSets ds = new DisjointUnionSets(size);
            for (LinkedList<Node> ln : graph) esize += (ln.size()-1);
            Edge[] edges = new Edge [esize/2];
            // The graph contains an array of list<Node>
            // For each list<Node>, it starts from the node itself and it is followed by its adjacent nodes
            // Using the starting node and the following nodes to generate edges
            for (LinkedList<Node> ln : graph) {
                for (int i = 1; i< ln.size(); i++) {
                    edges[edge_counter] = new Edge(ln.get(0),ln.get(i));
                    graph[ln.get(i).id].remove(ln.get(0)); // remove the symmetric egdes
                    edge_counter++;
                }
            }
            Arrays.sort(edges); // the sorting for arrays in java by default is quick sorting
            for (int i = 0; i<edge_counter;i++) {
                if (mst.size()>=size - 1) break;  // number of edges in mst must not exceed number of nodes
                Node v = edges[i].v;
                int parent1 = ds.find(v.id);
                Node u = edges[i].u;
                int parent2 = ds.find(u.id);
                // use Disjointset to build the MST
                if (parent1!=parent2) {
                    ds.union(v.id, u.id);
                    mst.add(edges[i]);
                }
            }
        }

    }
    // Using Prim algorithm to build MST
    public static class PrimMST {
        public List<Edge> mst = new ArrayList<Edge>();
        public double weight() {
            double weight = 0;
            for (Edge e: mst) weight+=e.weight;
            return weight;
        }

        // Given a random connected undirected graph, generate a MST
        // The graph is in linked list representation
        // Constructor for linked list graph
        public PrimMST(LinkedList<Node>[] graph) {
            int edge_counter = 0;
            int size = graph.length;
            Node[] nodes = new Node[size];
            for (int i = 0; i<size; i++) nodes[i] = graph[i].get(0);
            int esize = 0;
            for (LinkedList<Node> ln : graph) esize += (ln.size()-1);
            Edge[] edges = new Edge [esize/2];
            // The graph contains an array of list<Node>
            // For each list<Node>, it starts from the node itself and it is followed by its adjacent nodes
            // Using the starting node and the following nodes to generate edges
            for (LinkedList<Node> ln : graph) {
                for (int i = 1; i< ln.size(); i++) {
                    edges[edge_counter] = new Edge(ln.get(0),ln.get(i));
                    graph[ln.get(i).id].remove(ln.get(0)); // remove the symmetric egdes
                    edge_counter++;
                }
            }
            nodes[0].key=0; // set the first node as the starting node of the MST
            PriorityQueue<Node> queue = new PriorityQueue<>();
            HashMap<Node, Edge> neMap = new HashMap<>(); // used to store the corresponding edge for nodes (might be changed)
            for (Node n : nodes) queue.add(n);
            while (!queue.isEmpty()) {
                Node minNode = queue.poll();
                Edge eFromneMap = neMap.get(minNode);
                if (eFromneMap!=null) mst.add(eFromneMap);
                for (Edge e : edges) {
                    if (e.contain(minNode)) {
                        int adj = e.getother(minNode).id;
                        // find the adjacent nodes. If a node is in the queue and its key is less than the corresponding edge's weight
                        // then its key would be changed and hashmap would be updated
                        // The priority queue would also be updated
                        if (queue.contains(nodes[adj])&&e.weight<nodes[adj].key) {
                            nodes[adj].key = e.weight;
                            neMap.put(nodes[adj],e);
                            queue.remove(nodes[adj]);
                            queue.add(nodes[adj]);
                        }
                    }
                }
            }
        }
        }

        // Given the number of the nodes, generate a complete graph in symmetric matrix representation
    public static Edge[][] completegraph_matrix(int size) {
        Node[] nodes = new Node[size];
        Random rand = new Random();
        Edge[][] graph = new Edge[size][size];
        for (int i = 0; i < size; i++) {
            double a = rand.nextDouble();
            double b = rand.nextDouble();
            nodes[i] = new Node(a,b,i); // generate a random node, which is inside 1X1 square
        }
        for (int i = 0; i < size; i++) {
            for (int j = i ; j < size; j++) {
                    graph[i][j] = new Edge(nodes[i],nodes[j]);
                    graph[j][i] = graph[i][j]; // the matrix is symmetric
                }
            }
            return graph;
        }

        // Given the number of the nodes, generate a random connected graph in linked list representation
    public static LinkedList<Node>[] connectedgraph_list(int size) {
        Node[] nodes = new Node[size];
        Random rand = new Random();
        LinkedList<Node> graph[] = new LinkedList[size];
        for (int i=0; i<size; i++) graph[i] = new LinkedList(); // avoid nullPointException
        for (int i = 0; i < size; i++) {
            double a = rand.nextDouble();
            double b = rand.nextDouble();
            nodes[i] = new Node(a,b,i); // generate a random node, which is inside the 1X1 square
        }
        List<Integer> current = new ArrayList<>(); // Use to represent if the graph is connected
        // random select two nodes and assign an edge between them with the weight of their distance value
        while (current.size()!=size) {
            current.clear();
            int n1 =  rand.nextInt(size);
            int n2 =  rand.nextInt(size);
            while (n1==n2) n2 = rand.nextInt(size);
            if (graph[n1].size()==0) {graph[n1].add(nodes[n1]);}
            if (graph[n2].size()==0) {graph[n2].add(nodes[n2]);} // The first element of linkedlist is the node itself
            if ((!graph[n1].contains(nodes[n2]))&&(!graph[n2].contains(nodes[n1]))){
                // if the linkedlists do not contain the nodes, the nodes would be added to the lists respectively
                graph[n1].add(nodes[n2]);
                graph[n2].add(nodes[n1]);
                // Use BFS to check if the graph is connected
                boolean visited[] = new boolean[size]; // use to mark down nodes which has been visited
                LinkedList<Integer> linkedNodes = new LinkedList<>(); // use to link the nodes
                int linking_node = nodes[0].id; // take the first node as the starting node
                visited[linking_node]=true;
                linkedNodes.add(linking_node);
                while (linkedNodes.size() != 0)
                {
                    linking_node = linkedNodes.poll();
                    current.add(linking_node); // use to check if the graph is connected
                    for (Node n : graph[linking_node]) {
                        // go through each linkedlist
                        // find out which node has not been visited.
                        // If find some, then add them to the linkednodes and update visited[]
                        if (!visited[n.id]) {
                            visited[n.id] = true;
                            linkedNodes.add(n.id);
                        }
                    }
                }
            }
        }
        return graph;
    }


    private static class DisjointUnionSets {
        int[] rank;
        int[] parent;
        int size;

        public DisjointUnionSets(int size) {
            this.size = size;
            rank = new int[size];
            parent = new int[size];
            for (int i=0; i<size; i++) {
                // initialise each node, as in the parent of the node is the node itself and the rank is 0;
                parent[i] = i;
                rank[i] = 0;
            }
        }

        int find(int n) {
            // recursively find out the root
            if (parent[n]!=n) parent[n] = find(parent[n]);
            return parent[n];
        }

        void union(int n1, int n2) {
            // union two set based on rank
            int root1 = find(n1), root2 = find(n2);
            if (root1 != root2) {
                if (rank[root1] < rank[root2]) parent[root1] = root2;
                else if (rank[root1] > rank[root2]) parent[root2] = root1;
                else {
                    parent[root2] = root1;
                    rank[root1]++;
                }
            }
        }
    }

    private static class Node implements Comparable<Node>{
        public final double x;
        public final double y;
        public final int id;
        public double key;
        public Node (double x, double y, int id) {
            // each node is inside the 1X1 square
            if (x < 0 || x > 1) throw new IndexOutOfBoundsException("vertex out of bound");
            this.x = x;
            if (y < 0 || x > 1) throw new IndexOutOfBoundsException("vertex out of bound");
            this.y = y;
            this.id = id;
            key = Integer.MAX_VALUE; // the key for each node initially is set to infinity
                                     // the key is used to build the priority queue
        }

        @Override
        public int compareTo(Node o) {
            if      (this.key < o.key) return -1;
            else if (this.key > o.key) return +1;
            else return 0;
        }
    }

        private static class Edge implements Comparable<Edge> {
        public final Node u;
        public final Node v;
        public final double weight;
        public Edge(Node u, Node v) {
            this.u = u;
            this.v = v;
            weight = Math.sqrt(Math.pow(u.x-v.x,2) + Math.pow(u.y-v.y,2)); // the weight of the edge is the Euclidean distance
        }

        public boolean contain (Node n) {
            return u.id==n.id||v.id==n.id;
        }

        public Node getother (Node n) {
            if (n.equals(u)) return v;
            else if (n.equals(v)) return u;
            else return null;
        }

            @Override
            public int compareTo(Edge o) {
                if      (this.weight < o.weight) return -1;
                else if (this.weight > o.weight) return +1;
                else return 0;
            }
        }

    public static void main(String[] args) {
        int n[] = {100,500,1000,5000}; // different n
        int p = 50;
        int q = 50;

        // average time and average weight sum for the MST of a complete graph using Kruskal's algorithm
        for (int aN : n) {
            double weightsum = 0;
            long time = 0;
            for (int j = 0; j < p; j++) {
                Edge[][] graph = MST.completegraph_matrix(aN);
                long start = System.nanoTime();
                KruskalMST km = new KruskalMST(graph);
                weightsum += km.weight();
                long end = System.nanoTime();
                time += (end - start);
            }
            System.out.println("n = " + aN + "; graph: complete graph; algorithm: Kruskal; averageTime: "
                    + (time / p) + "ns; averageWeight: " + (weightsum / p));
        }
        System.out.println(" ");
        System.out.println(" ");
        // average time and average weight sum for the MST of a random connected graph using Kruskal's algorithm and Prim's algorithm
        for (int aN : n) {
            double weightsum_K = 0;
            long time_K = 0;
            double weightsum_P = 0;
            long time_P = 0;
            for (int j = 0; j < q; j++) {
                LinkedList<Node>[] graph = MST.connectedgraph_list(aN);
                long start = System.nanoTime();
                KruskalMST km = new KruskalMST(graph);
                weightsum_K += km.weight();
                long end = System.nanoTime();
                time_K += (end - start);

                graph = MST.connectedgraph_list(aN);
                start = System.nanoTime();
                PrimMST pm = new PrimMST(graph);
                weightsum_P += pm.weight();
                end = System.nanoTime();
                time_P += (end - start);
            }
            System.out.println("n = " + aN + "; graph: random connected graph; algorithm: Kruskal; averageTime: "
                    + (time_K / q) + "ns; averageWeight: " + (weightsum_K / q));
            System.out.println("n = " + aN + "; graph: random connected graph; algorithm: Prim; averageTime: "
                    + (time_P / q) + "ns; averageWeight: " + (weightsum_P / q));
        }

    }
}


