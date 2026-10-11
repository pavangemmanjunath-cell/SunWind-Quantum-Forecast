import { useMemo, useRef, type MutableRefObject } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Html, OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { REGION_COLOR, panelColor } from '../../lib/colors';

export interface RegionState { ghi: number; wind: number; stormy: boolean; cf: number }
export type StateRef = MutableRefObject<Record<string, RegionState>>;

/** Rotor speed follows the turbine regime: idle below cut-in (3 m/s), proportional up to rated (12 m/s), parked at cut-out (25 m/s). */
const rotorSpeed = (w: number): number => (w < 3 || w >= 25 ? 0 : Math.min(w, 12) * 0.22);

function Turbine({ position, scale = 1, region, stateRef }: { position: [number, number, number]; scale?: number; region: string; stateRef: StateRef }) {
  const rotor = useRef<THREE.Group>(null);
  useFrame((_, dt) => {
    if (rotor.current) rotor.current.rotation.z -= rotorSpeed(stateRef.current[region]?.wind ?? 0) * dt;
  });
  return (
    <group position={position} scale={scale}>
      <mesh position={[0, 2, 0]} castShadow>
        <cylinderGeometry args={[0.06, 0.13, 4, 12]} />
        <meshStandardMaterial color="#e6ebff" roughness={0.6} />
      </mesh>
      <mesh position={[0, 4.05, 0]}>
        <boxGeometry args={[0.25, 0.25, 0.55]} />
        <meshStandardMaterial color="#cbd5f5" />
      </mesh>
      <group ref={rotor} position={[0, 4.05, 0.34]}>
        <mesh>
          <sphereGeometry args={[0.13, 14, 14]} />
          <meshStandardMaterial color="#94a3d8" />
        </mesh>
        {[0, 1, 2].map((i) => (
          <group key={i} rotation={[0, 0, (i * 2 * Math.PI) / 3]}>
            <mesh position={[0, 0.95, 0]}>
              <boxGeometry args={[0.12, 1.9, 0.025]} />
              <meshStandardMaterial color="#f1f5ff" />
            </mesh>
          </group>
        ))}
      </group>
    </group>
  );
}

function SolarField({ origin, rows, cols, region, stateRef }: { origin: [number, number, number]; rows: number; cols: number; region: string; stateRef: StateRef }) {
  const mat = useMemo(() => new THREE.MeshStandardMaterial({ color: '#0b2a5b', metalness: 0.5, roughness: 0.25 }), []);
  const sun = useRef<THREE.MeshStandardMaterial>(null);
  useFrame(() => {
    const g = stateRef.current[region]?.ghi ?? 0;
    mat.color.copy(panelColor(g));
    mat.emissive.copy(panelColor(g)).multiplyScalar(0.25);
    if (sun.current) sun.current.emissiveIntensity = 0.15 + g / 700;
  });
  const items: [number, number][] = [];
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) items.push([c * 1.15, r * 1.05]);
  return (
    <group position={origin}>
      {items.map(([x, z], i) => (
        <mesh key={i} position={[x, 0.35, z]} rotation={[-0.55, 0, 0]} material={mat}>
          <boxGeometry args={[1.0, 0.04, 0.8]} />
        </mesh>
      ))}
      <mesh position={[(cols * 1.15) / 2, 6.4, -1]}>
        <sphereGeometry args={[0.55, 24, 24]} />
        <meshStandardMaterial ref={sun} color="#fde68a" emissive="#fbbf24" emissiveIntensity={0.3} />
      </mesh>
    </group>
  );
}

function Storm({ position, region, stateRef }: { position: [number, number, number]; region: string; stateRef: StateRef }) {
  const grp = useRef<THREE.Group>(null);
  const light = useRef<THREE.PointLight>(null);
  const s = useRef(0);
  useFrame((st, dt) => {
    const on = stateRef.current[region]?.stormy ? 1 : 0;
    s.current += (on - s.current) * Math.min(1, dt * 2.5);
    if (grp.current) {
      grp.current.scale.setScalar(Math.max(0.001, s.current));
      grp.current.position.y = position[1] + Math.sin(st.clock.elapsedTime * 0.6) * 0.08;
    }
    if (light.current) light.current.intensity = s.current > 0.6 && Math.random() > 0.93 ? 6 : 0;
  });
  const blobs: [number, number, number, number][] = [[0, 0, 0, 1.4], [1.5, -0.2, 0.3, 1.1], [-1.5, -0.1, -0.2, 1.2], [0.6, 0.4, -0.7, 1.0], [-0.7, 0.3, 0.6, 0.9]];
  return (
    <group ref={grp} position={position}>
      {blobs.map(([x, y, z, r], i) => (
        <mesh key={i} position={[x, y, z]}>
          <sphereGeometry args={[r, 18, 18]} />
          <meshStandardMaterial color="#3a4468" roughness={1} transparent opacity={0.88} />
        </mesh>
      ))}
      <pointLight ref={light} color="#bcd6ff" intensity={0} distance={14} position={[0, -0.8, 0]} />
    </group>
  );
}

function Platform({ x, region, label, selected }: { x: number; region: string; label: string; selected: boolean }) {
  return (
    <group position={[x, 0, 0]}>
      <mesh position={[0, -0.1, 1.2]} receiveShadow>
        <boxGeometry args={[8, 0.2, 7]} />
        <meshStandardMaterial color={selected ? '#1d2a5c' : '#141c3c'} emissive={REGION_COLOR[region]} emissiveIntensity={selected ? 0.18 : 0.04} />
      </mesh>
      <Html position={[0, 0.2, 5.3]} center distanceFactor={14}>
        <div className="hover-label" style={{ transform: 'none', borderColor: REGION_COLOR[region] }}>
          {region} · {label}
        </div>
      </Html>
    </group>
  );
}

interface Props { stateRef: StateRef; selected: string; names: Record<string, string> }

export default function RegionsScene({ stateRef, selected, names }: Props) {
  return (
    <Canvas shadows camera={{ position: [0, 11, 19], fov: 45 }} dpr={[1, 2]}>
      <color attach="background" args={['#0b1020']} />
      <fog attach="fog" args={['#0b1020', 30, 70]} />
      <ambientLight intensity={0.7} />
      <directionalLight position={[8, 14, 6]} intensity={1.1} castShadow />
      <gridHelper args={[60, 60, '#27325c', '#1b2547']} position={[0, -0.21, 0]} />

      <Platform x={-9} region="A" label={names.A} selected={selected === 'A'} />
      <SolarField origin={[-12, 0, -0.6]} rows={5} cols={5} region="A" stateRef={stateRef} />
      <Turbine position={[-6.2, 0, 1.2]} scale={0.8} region="A" stateRef={stateRef} />
      <Storm position={[-9, 6.8, 0]} region="A" stateRef={stateRef} />

      <Platform x={0} region="B" label={names.B} selected={selected === 'B'} />
      {[-2.7, -0.9, 0.9, 2.7].map((x, i) => (
        <Turbine key={x} position={[x, 0, i % 2 === 0 ? 0 : 2.3]} region="B" stateRef={stateRef} />
      ))}
      <Storm position={[0, 7.4, 0]} region="B" stateRef={stateRef} />

      <Platform x={9} region="C" label={names.C} selected={selected === 'C'} />
      <SolarField origin={[6.2, 0, -0.6]} rows={4} cols={3} region="C" stateRef={stateRef} />
      <Turbine position={[10.7, 0, 0.4]} scale={0.9} region="C" stateRef={stateRef} />
      <Turbine position={[12.4, 0, 2.4]} scale={0.9} region="C" stateRef={stateRef} />
      <Storm position={[9, 7.2, 0]} region="C" stateRef={stateRef} />

      <OrbitControls enablePan={false} minDistance={8} maxDistance={40} maxPolarAngle={Math.PI / 2.1} target={[0, 2, 1]} />
    </Canvas>
  );
}
