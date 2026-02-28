export default function Skeleton() {
  return (
    <div className="rounded-2xl border border-line bg-surface/65 p-4">
      <div className="h-4 w-20 rounded bg-[linear-gradient(110deg,#eadfd5,40%,#f8f2eb,50%,#eadfd5,60%)] bg-[length:200%_100%] animate-shimmer" />
      <div className="mt-4 space-y-2">
        <div className="h-3 w-full rounded bg-[linear-gradient(110deg,#eadfd5,40%,#f8f2eb,50%,#eadfd5,60%)] bg-[length:200%_100%] animate-shimmer" />
        <div className="h-3 w-[90%] rounded bg-[linear-gradient(110deg,#eadfd5,40%,#f8f2eb,50%,#eadfd5,60%)] bg-[length:200%_100%] animate-shimmer" />
        <div className="h-3 w-[75%] rounded bg-[linear-gradient(110deg,#eadfd5,40%,#f8f2eb,50%,#eadfd5,60%)] bg-[length:200%_100%] animate-shimmer" />
      </div>
    </div>
  );
}
