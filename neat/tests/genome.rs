#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use new_york_utils::Matrix;
    use serde_json::json;

    use vivalaakam_neuro_neat::{Config, Connection, Genome, NeuronType, Node};
    use vivalaakam_neuro_utils::Activation;

    #[test]
    fn it_works() {
        let nodes = vec![
            Node::new(NeuronType::Input, 0, 0f32, None, None),
            Node::new(NeuronType::Hidden, 2, 0f32, None, None),
            Node::new(NeuronType::Hidden, 3, 0f32, None, None),
            Node::new(NeuronType::Output, 1, 0f32, None, None),
        ];
        let connections = vec![
            Connection::new(0, 1, 0f32),
            Connection::new(0, 2, 0f32),
            Connection::new(0, 3, 0f32),
            Connection::new(2, 1, 0f32),
            Connection::new(3, 1, 0f32),
        ];

        let genome = Genome::new(nodes, connections);

        assert_eq!(genome.get_nodes().len(), 4);
        assert_eq!(genome.get_connections().len(), 5);
        assert_eq!(
            json!(genome).to_string(),
            r#"{"connections":[{"enabled":true,"from":0,"to":1,"weight":0.0},{"enabled":true,"from":0,"to":2,"weight":0.0},{"enabled":true,"from":0,"to":3,"weight":0.0},{"enabled":true,"from":2,"to":1,"weight":0.0},{"enabled":true,"from":3,"to":1,"weight":0.0}],"nodes":[{"activation":"Identity","bias":0.0,"enabled":true,"id":0,"neuron_type":"Input","position":0},{"activation":"Identity","bias":0.0,"enabled":true,"id":2,"neuron_type":"Hidden","position":1},{"activation":"Identity","bias":0.0,"enabled":true,"id":3,"neuron_type":"Hidden","position":2},{"activation":"Identity","bias":0.0,"enabled":true,"id":1,"neuron_type":"Output","position":3}]}"#
        );
    }

    #[test]
    fn from_string() {
        let string = r#"{"connections":[{"enabled":true,"from":0,"to":1,"weight":0.0},{"enabled":true,"from":0,"to":2,"weight":0.0},{"enabled":true,"from":0,"to":3,"weight":0.0},{"enabled":true,"from":2,"to":1,"weight":0.0},{"enabled":true,"from":3,"to":1,"weight":0.0}],"nodes":[{"activation":"Identity","bias":0.0,"enabled":true,"id":0,"neuron_type":"Input","position":0},{"activation":"Identity","bias":0.0,"enabled":true,"id":2,"neuron_type":"Hidden","position":1},{"activation":"Identity","bias":0.0,"enabled":true,"id":3,"neuron_type":"Hidden","position":2},{"activation":"Identity","bias":0.0,"enabled":true,"id":1,"neuron_type":"Output","position":3}]}"#;
        let genome: Genome = string.to_string().into();

        assert_eq!(genome.get_nodes().len(), 4);
        assert_eq!(genome.get_connections().len(), 5);
    }

    #[test]
    fn from_str() {
        let string = r#"{"connections":[{"enabled":true,"from":0,"to":1,"weight":0.0},{"enabled":true,"from":0,"to":2,"weight":0.0},{"enabled":true,"from":0,"to":3,"weight":0.0},{"enabled":true,"from":2,"to":1,"weight":0.0},{"enabled":true,"from":3,"to":1,"weight":0.0}],"nodes":[{"activation":"Identity","bias":0.0,"enabled":true,"id":0,"neuron_type":"Input","position":0},{"activation":"Identity","bias":0.0,"enabled":true,"id":2,"neuron_type":"Hidden","position":1},{"activation":"Identity","bias":0.0,"enabled":true,"id":3,"neuron_type":"Hidden","position":2},{"activation":"Identity","bias":0.0,"enabled":true,"id":1,"neuron_type":"Output","position":3}]}"#;
        let genome: Genome = string.into();

        assert_eq!(genome.get_nodes().len(), 4);
        assert_eq!(genome.get_connections().len(), 5);
    }

    #[test]
    fn generate_genome() {
        let config = Config {
            node_bias: 1.0,
            connection_weight: 1.0,
            ..Config::default()
        };

        let genome = Genome::generate_genome(1, 1, vec![2], None, &config);

        assert_eq!(genome.get_nodes().len(), 4);
        assert_eq!(genome.get_connections().len(), 4);
    }

    #[test]
    fn get_network() {
        let nodes = vec![
            Node::new(NeuronType::Input, 0, 0.0, None, Some(1)),
            Node::new(
                NeuronType::Hidden,
                2,
                0.5,
                Some(Activation::Sigmoid),
                Some(2),
            ),
            Node::new(
                NeuronType::Hidden,
                3,
                0.4,
                Some(Activation::Sigmoid),
                Some(3),
            ),
            Node::new(
                NeuronType::Output,
                1,
                0.3,
                Some(Activation::Sigmoid),
                Some(4),
            ),
        ];
        let connections = vec![
            Connection::new(0, 1, 0.9),
            Connection::new(0, 2, 0.7),
            Connection::new(0, 3, 0.5),
            Connection::new(2, 1, 0.3),
            Connection::new(3, 1, 0.1),
        ];

        let genome = Genome::new(nodes, connections);

        let network = genome.get_network();

        assert_eq!(network.activate(vec![1.0]), vec![0.9996177]);
        assert_eq!(network.activate(vec![1.2]), vec![0.9998430]);
        assert_eq!(network.activate(vec![0.5]), vec![0.99639386]);
        assert_eq!(network.activate(vec![0.1]), vec![0.977193]);
    }

    #[test]
    fn get_network_matrix() {
        let nodes = vec![
            Node::new(NeuronType::Input, 0, 0.0, None, Some(1)),
            Node::new(
                NeuronType::Hidden,
                2,
                0.5,
                Some(Activation::Sigmoid),
                Some(2),
            ),
            Node::new(
                NeuronType::Hidden,
                3,
                0.4,
                Some(Activation::Sigmoid),
                Some(3),
            ),
            Node::new(
                NeuronType::Output,
                1,
                0.3,
                Some(Activation::Sigmoid),
                Some(4),
            ),
        ];
        let connections = vec![
            Connection::new(0, 1, 0.9),
            Connection::new(0, 2, 0.7),
            Connection::new(0, 3, 0.5),
            Connection::new(2, 1, 0.3),
            Connection::new(3, 1, 0.1),
        ];

        let genome = Genome::new(nodes, connections);

        let network = genome.get_network();

        let mut inputs = Matrix::new(1, 4);
        let _ = inputs.set_data(vec![1.0, 1.2, 0.5, 0.1]);

        let inputs = Array2::from_shape_vec((4, 1), vec![1.0, 1.2, 0.5, 0.1]).expect("input error");
        let outputs =
            Array2::from_shape_vec((4, 1), vec![0.9996177, 0.9998430, 0.99639386, 0.977193])
                .expect("output error");

        assert_eq!(
            format!("{:?}", network.activate_matrix(&inputs)),
            format!("{:?}", outputs)
        );
    }

    #[test]
    fn add_node() {
        let config = Config {
            node_bias: 1.0,
            node_max: 10,
            ..Config::default()
        };

        let nodes = vec![
            Node::new(NeuronType::Input, 0, 0.0, None, Some(1)),
            Node::new(NeuronType::Output, 1, -0.3, None, Some(2)),
        ];
        let connections = vec![Connection::new(0, 1, 0.7)];

        let genome = Genome::new(nodes, connections);
        let new_genome = genome.mutate_add_node(&config).unwrap();

        assert_eq!(new_genome.get_nodes().len(), 3);
        assert_eq!(new_genome.get_connections().len(), 3);
    }

    #[test]
    fn mutate_connection_weight() {
        let config = Config {
            connection_weight_delta: 1.0,
            ..Config::default()
        };

        let nodes = vec![
            Node::new(NeuronType::Input, 0, 0f32, None, None),
            Node::new(NeuronType::Output, 1, -0.3f32, None, None),
        ];
        let connections = vec![Connection::new(0, 1, 0.7f32)];

        let genome = Genome::new(nodes, connections);
        let new_genome = genome.mutate_connection_weight(&config).unwrap();

        assert_ne!(
            new_genome.get_connections().first().unwrap().get_weight(),
            0.7f32
        );
    }

    #[test]
    fn mutate_connection_enabled() {
        let nodes = vec![
            Node::new(NeuronType::Input, 0, 0f32, None, None),
            Node::new(NeuronType::Input, 1, 0f32, None, None),
            Node::new(NeuronType::Output, 2, 0f32, None, None),
        ];
        let connections = vec![Connection::new(0, 2, 0f32), Connection::new(1, 2, 0f32)];

        let genome = Genome::new(nodes, connections);
        let new_genome = genome.mutate_connection_enabled().unwrap();

        let mut enabled = 0;
        let mut disabled = 0;

        for con in new_genome.get_connections() {
            if con.get_enabled() {
                enabled += 1;
            } else {
                disabled += 1;
            }
        }

        assert_eq!(enabled, 1);
        assert_eq!(disabled, 1);
    }

    #[test]
    fn mutate_crossover() {
        let nodes = vec![
            Node::new(NeuronType::Input, 0, 0.0, None, None),
            Node::new(NeuronType::Input, 1, 0.0, None, None),
            Node::new(NeuronType::Hidden, 2, 0.0, None, None),
            Node::new(NeuronType::Output, 3, 0.0, None, None),
        ];
        let connections = vec![
            Connection::new(0, 3, 0.0),
            Connection::new(1, 3, 0.0),
            Connection::new(0, 2, 0.0),
            Connection::new(2, 3, 0.0),
        ];

        let genome = Genome::new(nodes, connections);

        let nodes = vec![
            Node::new(NeuronType::Input, 0, 0.0, None, None),
            Node::new(NeuronType::Input, 1, 0.0, None, None),
            Node::new(NeuronType::Hidden, 3, 0.0, None, None),
            Node::new(NeuronType::Output, 4, 0.0, None, None),
        ];
        let connections = vec![
            Connection::new(0, 4, 0.0),
            Connection::new(1, 4, 0.0),
            Connection::new(1, 3, 0.0),
            Connection::new(3, 4, 0.0),
        ];

        let child_genome = Genome::new(nodes, connections);

        let new_genome = genome.mutate_crossover(&child_genome).unwrap();

        assert_eq!(new_genome.get_connections().len(), 7);
        assert_eq!(new_genome.get_nodes().len(), 5);
    }

    #[ignore]
    #[test]
    fn mutate_node_enabled() {
        let config = Config::default();

        let nodes = vec![
            Node::new(NeuronType::Input, 0, 0f32, None, None),
            Node::new(NeuronType::Hidden, 2, 0.75f32, Some(Activation::Relu), None),
            Node::new(NeuronType::Output, 1, 2f32, Some(Activation::Relu), None),
        ];
        let connections = vec![
            Connection::new(0, 1, 1.5f32),
            Connection::new(0, 2, 2f32),
            Connection::new(2, 1, 3f32),
        ];

        let genome = Genome::new(nodes, connections);
        let network = genome.get_network();
        assert_eq!(network.activate(vec![1.0]), vec![11.75]);

        match genome.mutate_node_enabled(&config) {
            None => {}
            Some(new_genome) => {
                let network = new_genome.get_network();
                assert_eq!(network.activate(vec![1.0]), vec![3.5]);
            }
        }
    }

    #[test]
    fn get_distance() {
        let nodes = vec![
            Node::new(NeuronType::Input, 0, 0.0, None, None),
            Node::new(NeuronType::Input, 1, 0.0, None, None),
            Node::new(NeuronType::Hidden, 2, 0.0, None, None),
            Node::new(NeuronType::Output, 3, 0.0, None, None),
        ];
        let connections = vec![
            Connection::new(0, 3, 0.0),
            Connection::new(1, 3, 0.0),
            Connection::new(0, 2, 0.0),
            Connection::new(2, 3, 0.0),
        ];

        let genome = Genome::new(nodes, connections);

        let nodes = vec![
            Node::new(NeuronType::Input, 0, 0.0, None, None),
            Node::new(NeuronType::Input, 1, 0.0, None, None),
            Node::new(NeuronType::Hidden, 2, 0.0, None, None),
            Node::new(NeuronType::Output, 3, 0.0, None, None),
        ];
        let connections = vec![
            Connection::new(0, 3, 0.0),
            Connection::new(1, 3, 0.0),
            Connection::new(1, 2, 0.0),
            Connection::new(2, 3, 0.0),
        ];

        let child_genome = Genome::new(nodes, connections);

        assert_eq!(genome.get_distance(&child_genome), 0);

        let nodes = vec![
            Node::new(NeuronType::Input, 0, 0.0, None, None),
            Node::new(NeuronType::Input, 1, 0.0, None, None),
            Node::new(NeuronType::Hidden, 2, 0.0, None, None),
            Node::new(NeuronType::Hidden, 3, 0.0, None, None),
            Node::new(NeuronType::Output, 4, 0.0, None, None),
        ];
        let connections = vec![
            Connection::new(0, 4, 0.0),
            Connection::new(1, 4, 0.0),
            Connection::new(1, 2, 0.0),
            Connection::new(2, 4, 0.0),
            Connection::new(1, 3, 0.0),
            Connection::new(3, 4, 0.0),
        ];

        let child_genome = Genome::new(nodes, connections);

        assert_eq!(genome.get_distance(&child_genome), 1);
    }
}
