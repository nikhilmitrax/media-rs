//! Deterministic algebraic and memory-safety properties over broad metadata.
use media_core::{
    media::{ImageLayout, PixelFormat, PlaneLayout},
    storage::StorageLease,
    time::Time,
};

fn draw(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn rational_time_preserves_algebraic_identities_at_random_and_extreme_values() {
    let mut state = 0x1278_3321_9074_8123;
    for i in 0..10_000 {
        let numerator = match i % 13 {
            0 => i64::MIN,
            1 => i64::MAX,
            2 => 0,
            _ => draw(&mut state) as i64,
        };
        let denominator = match i % 7 {
            0 => u32::MAX,
            1 => 1,
            2 => 48000,
            _ => (draw(&mut state) as u32).max(1),
        };
        let time = Time::new(numerator, denominator).unwrap();
        assert_eq!(time.checked_add(Time::ZERO).unwrap(), time);
        assert_eq!(time.checked_sub(time).unwrap(), Time::ZERO);
        assert_eq!(time.checked_mul_ratio(0, 1).unwrap(), Time::ZERO);
        assert_eq!(
            Time::new(time.numerator(), time.denominator()).unwrap(),
            time
        );
        let other = Time::new((draw(&mut state) % 100_000) as i64 - 50_000, 1000).unwrap();
        if let Ok(sum) = time.checked_add(other) {
            assert_eq!(sum.checked_sub(other).unwrap(), time);
        }
        assert_eq!(time.cmp(&other), other.cmp(&time).reverse());
    }
}

#[test]
fn accepted_random_pitched_layouts_always_address_real_visible_rows() {
    let mut state = 0x7754_1880_a912_b6c4;
    for i in 0..10_000 {
        let width = (draw(&mut state) % 32 + 1) as u32;
        let height = (draw(&mut state) % 32 + 1) as u32;
        let row = width as usize * 3;
        let stride = row + (draw(&mut state) % 16) as usize;
        let offset = (draw(&mut state) % 64) as usize;
        let length = offset + stride * (height as usize - 1) + row;
        let storage = StorageLease::from_bytes(vec![0; length]);
        let mut plane = PlaneLayout {
            offset,
            row_stride: stride,
            row_bytes: row,
            rows: height as usize,
        };
        match i % 5 {
            0 => plane.row_stride = usize::MAX,
            1 => plane.offset = usize::MAX,
            2 => plane.rows = usize::MAX,
            _ => {}
        }
        let layout = ImageLayout {
            width,
            height,
            allocated_width: width,
            allocated_height: height,
            format: PixelFormat::Rgb8,
            planes: vec![plane],
        };
        if layout.validate(&storage).is_ok() {
            let plane = &layout.planes[0];
            for y in 0..height as usize {
                let begin = plane
                    .offset
                    .checked_add(y.checked_mul(plane.row_stride).unwrap())
                    .unwrap();
                let end = begin.checked_add(row).unwrap();
                assert!(storage.host_bytes().unwrap().get(begin..end).is_some());
            }
        }
    }
}
