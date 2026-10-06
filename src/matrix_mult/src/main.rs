use std::cmp::min;
use std::io::{self, Write};
use std::time::Instant;

fn on_mult(m_ar: usize, m_br: usize) {

    let pha = vec![1.0f64; m_ar * m_ar];
    let mut phb = vec![0.0f64; m_ar * m_br];
    let mut phc = vec![0.0f64; m_ar * m_br];

    for i in 0..m_ar {
        for j in 0..m_br {
            phb[i * m_br + j] = (i + 1) as f64;
        }
    }

    let start = Instant::now();

    for i in 0..m_ar {
        for j in 0..m_br {
            let mut temp = 0.0;
            for k in 0..m_ar {
                temp += pha[i * m_ar + k] * phb[k * m_br + j];
            }
            phc[i * m_br + j] = temp;
        }
    }

    let duration = start.elapsed();
    println!("Time: {:.3} seconds", duration.as_secs_f64());
    println!("Result matrix: ");

    for j in 0..min(10, m_br) {
        print!("{} ", phc[j]);
    }
    println!();
}

// Line-by-line matrix multiplication
fn on_mult_line(m_ar: usize, m_br: usize) {
    let pha = vec![1.0f64; m_ar * m_ar];
    let mut phb = vec![0.0f64; m_ar * m_br];
    let mut phc = vec![0.0f64; m_ar * m_br]; // Automatically initialized to 0.0

    for i in 0..m_ar {
        for j in 0..m_br {
            phb[i * m_br + j] = (i + 1) as f64;
        }
    }

    let start = Instant::now();

    // Loop order: i, k, j
    for i in 0..m_ar {
        for k in 0..m_ar {
            for j in 0..m_br {
                phc[i * m_br + j] += pha[i * m_ar + k] * phb[k * m_br + j];
            }
        }
    }

    let duration = start.elapsed();
    println!("Time: {:.3} seconds", duration.as_secs_f64());
    println!("Result matrix: ");

    for j in 0..min(10, m_br) {
        print!("{} ", phc[j]);
    }
    println!();
}

// Block matrix multiplication
fn on_mult_block(m_ar: usize, m_br: usize, bk_size: usize) {
    let pha = vec![1.0f64; m_ar * m_ar];
    let mut phb = vec![0.0f64; m_ar * m_br];
    let mut phc = vec![0.0f64; m_ar * m_br]; // Automatically initialized to 0.0

    // Initialize phb
    for i in 0..m_ar {
        for j in 0..m_br {
            phb[i * m_br + j] = (i + 1) as f64;
        }
    }

    let start = Instant::now();

    for ii in (0..m_ar).step_by(bk_size) {
        for kk in (0..m_ar).step_by(bk_size) {
            for jj in (0..m_br).step_by(bk_size){
                for i in ii..min(ii + bk_size, m_ar) {
                    for k in kk..min(kk + bk_size, m_ar) {
                        for j in jj..min(jj + bk_size, m_br) {
                            phc[i * m_br + j] += pha[i * m_ar + k] * phb[k * m_br + j];
                        }
                    }
                }
            }
        }
    }

    let duration = start.elapsed();
    println!("Time: {:.3} seconds", duration.as_secs_f64());
    println!("Result matrix: ");

    for j in 0..min(10, m_br) {
        print!("{} ", phc[j]);
    }
    println!();
}

fn read_input(prompt: &str) -> usize {
    print!("{}", prompt);
    io::stdout().flush().unwrap();    
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    
    input.trim().parse().unwrap_or(0)
}

fn main() {
    loop {
        //println!("\n1. Multiplication");
        //println!("2. Line Multiplication");
        //println!("3. Block Multiplication");
        //println!("0. Exit");
        
        let op = read_input("Selection?: ");

        if op == 0 {
            break;
        }

        let lin = read_input("Dimensions: lins=cols ? ");
        let col = lin;

        match op {
            1 => on_mult(lin, col),
            2 => on_mult_line(lin, col),
            3 => {
                let block_size = read_input("Block Size? ");
                on_mult_block(lin, col, block_size);
            }
            _ => println!("Invalid selection, please try again."),
        }
    }
}
